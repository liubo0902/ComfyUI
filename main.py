import comfy.options
comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time
import shutil
import re

def execute_prestartup_script():
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            print(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        print("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()

execute_prestartup_script()


# Main code
import asyncio
import itertools
import shutil
import threading
import gc

from comfy.cli_args import args
import logging

if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc

import comfy.utils
import yaml

import execution
import server
from server import BinaryEventTypes
from nodes import init_custom_nodes
import comfy.model_management

def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")

def prompt_worker(q, server):
    e = execution.PromptExecutor(server)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server.last_prompt_id = prompt_id

            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            q.task_done(item_id,
                        e.outputs_ui,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))
            if server.client_id is not None:
                server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, server.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                comfy.model_management.cleanup_models()
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False

async def run(server, address='', port=8188, verbose=True, call_on_start=None):
    await asyncio.gather(server.start(address, port, verbose, call_on_start), server.publish_loop())


def hijack_progress(server):
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()
        progress = {"value": value, "max": total, "prompt_id": server.last_prompt_id, "node": server.last_node_id}

        server.send_sync("progress", progress, server.client_id)
        if preview_image is not None:
            server.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server.client_id)
    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                logging.info("Adding extra search path {} {}".format(x, full_path))
                folder_paths.add_model_folder_path(x, full_path)
def natural_sort_key(s, regex=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in regex.split(s)]

def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return

    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    items = list(os.walk(path, followlinks=True))
    items = sorted(items, key=lambda x: natural_sort_key(x[0]))

    for root, _, files in items:
        for filename in sorted(files, key=natural_sort_key):
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue

            yield os.path.join(root, filename)

def load_work_dir(data_dir):
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'outputs'), exist_ok=True)
            folder_paths.set_output_directory(os.path.join(data_dir, 'output'))
        except:
            pass
    if os.path.exists(os.path.join(data_dir, 'models/Stable-diffusion')):
        webui_dict = {
            'checkpoints': ['models/Stable-diffusion'], 
            'configs': ['models/Stable-diffusion'], 
            'vae': ['models/VAE'],
            'loras': ['models/Lora', 'models/LyCORIS'],
            'upscale_models': ['models/ESRGAN', 'models/RealESRGAN', 'models/SwinIR'], 
            'embeddings': ['embeddings'], 
            'hypernetworks': ['models/hypernetworks'], 
            'controlnet': ['models/ControlNet']
        }
        for key in webui_dict:
            for folder_name in webui_dict[key]:
                if os.path.exists(os.path.join(data_dir, folder_name)):
                    folder_paths.add_model_folder_path(key, os.path.join(data_dir, folder_name))
    if os.path.exists(os.path.join(data_dir, 'models/checkpoints')):
        cmui_dict = {
            'checkpoints': 'models/checkpoints', 
            'clip': 'models/clip',
            'clip_vision': 'models/clip_vision',
            'configs': 'models/configs',
            'controlnet': 'models/controlnet', 
            'embeddings': 'models/embeddings',
            'loras': 'models/loras',
            'upscale_models': 'models/upscale_models',
            'vae': 'models/vae'
        }
        for key in cmui_dict:
            folder_name = webui_dict[key]
            if os.path.exists(os.path.join(data_dir, folder_name)):
                folder_paths.add_model_folder_path(key, os.path.join(data_dir, folder_name))
    os.makedirs(os.path.join(data_dir, 'custom_nodes'), exist_ok=True)
    for custom_node_folder in os.listdir(os.path.join(data_dir, 'custom_nodes')):
        for filepath in walk_files(os.path.join(data_dir, 'custom_nodes', custom_node_folder)):
            folder = os.path.basename(filepath).replace(os.path.join(data_dir, 'custom_nodes'), 'custom_nodes')
            os.makedirs(folder, exist_ok=True)
            shutil.copyfile(filepath, filepath.replace(os.path.join(data_dir, 'custom_nodes'), 'custom_nodes'))

if __name__ == "__main__":
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = server.PromptServer(loop)
    q = execution.PromptQueue(server)

    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)
    if args.data_dir:
        load_work_dir(args.data_dir)

    init_custom_nodes()

    cuda_malloc_warning()

    server.add_routes()
    hijack_progress(server)

    threading.Thread(target=prompt_worker, daemon=True, args=(q, server,)).start()

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    #These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.quick_test_for_ci:
        exit(0)

    call_on_start = None
    if args.auto_launch:
        def startup_server(address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            webbrowser.open(f"http://{address}:{port}")
        call_on_start = startup_server

    try:
        loop.run_until_complete(run(server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start))
    except KeyboardInterrupt:
        logging.info("\nStopped server")

    cleanup_temp()
