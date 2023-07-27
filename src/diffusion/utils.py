import argparse
import io
import logging
import math
import tarfile
from typing import List

import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionAttendAndExcitePipeline,
    StableDiffusionPipeline,
    UnCLIPPipeline,
    UnCLIPScheduler,
)
from omegaconf import OmegaConf
from PIL import Image

from .deepfloyd.deepfloyd_pipeline import IF_pipeline_all


def get_torch_precision(precision: str):
    return torch.float32 if precision == "float32" else torch.float16


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

AVAILABLE_PIPELINE = [
    "SD",
    "attend_and_excite",
    "unclip",
    "IF",
]


def get_pipeline_diffusion(
    model_path, precision, pipeline="SD", requires_safety_checker=False
):
    if pipeline == "IF":
        return IF_pipeline_all(**model_path)
    elif pipeline == "SD":
        pipeline = StableDiffusionPipeline
    elif pipeline == "attend_and_excite":
        pipeline = StableDiffusionAttendAndExcitePipeline
    elif pipeline == "unclip":
        pipeline = UnCLIPPipeline
    return pipeline.from_pretrained(
        model_path,
        requires_safety_checker=requires_safety_checker,
        torch_dtype=get_torch_precision(precision),
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="configs/stable_inference.yaml",
        help="path to the config of the pipeline of the diffusion model",
        required=True,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        help=f"path to the tarfile where the images will be saved"
        f"if not indicated, use the path indicate in the config file. If only a path to a directory is indicated,"
        f"a tarball will be created in this directory with the names `images.tar`",
        required=True,
    )

    parser.add_argument(
        "--index_file",
        type=str,
        help="path to the index file, i.e. list of prompts that have been generated. Enable to restart the generation. If not indicated will create `index.txt` in the same directory as the tarfile",
        required=False,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="path to the dataset.",
        required=True,
    )

    parser.add_argument(
        "--precision",
        type=str,
        nargs="?",
        default="float16",
        choices=["float32", "float16"],
        help="precision  weights of the model, save GPU memory",
        required=False,
    )

    parser.add_argument(
        "--image_per_prompt",
        type=int,
        required=True,
        help="number of image to generate for each prompt",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch size for the generation",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite the existing experiment"
    )

    parser.add_argument(
        "--mps",
        action="store_true",
        help="to activate when on mps device",
    )

    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="From diffusers documentation : "
        "Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared"
        "to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`"
        "method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with"
        "`enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`."
        "This model is not available for Attend and Excite models.",
    )
    parser.add_argument("--attention_slice", type=int, required=False, default=None)

    parser.add_argument(
        "--vae_slicing",
        action="store_true",
    )
    return parser.parse_args()


def create_config(args, config):
    if "pipeline" in config.keys():
        if config.pipeline not in AVAILABLE_PIPELINE:
            raise ValueError(
                f"pipeline {config.pipeline} not available, available pipeline are {AVAILABLE_PIPELINE}"
            )
        else:
            pipeline = config.pipeline
    else:
        print("pipeline not indicated, use the  default SD diffusion pipeline")
        pipeline = "SD"

    if pipeline in [
        "SD",
        "attend_and_excite",
    ]:
        # Latent diffusion model
        if "image_size" in config.keys():
            height, width = config.image_size
        else:
            height, width = None, None
        if "inference_steps" in config.keys():
            inference_steps = config.inference_steps
        else:
            inference_steps = 50
        params_inference = {
            "height": height,
            "width": width,
            "num_inference_steps": inference_steps,
            "guidance_scale": config.guidance_scale,
        }
    else:
        params_inference = {}

    new_config = {
        "save_dir": args.save_dir,
        "precision": args.precision,
        "model_path": config.model_path,
        "dataset": args.dataset,
        "image_per_prompt": args.image_per_prompt,
        "dataset_path": args.dataset,
        "batch_size": args.batch_size,
        "overwrite": args.overwrite,
        "pipeline": pipeline,
        "scheduler": config.scheduler if "scheduler" in config.keys() else None,
        "index": args.index_file,
        "enable_model_cpu_offload": args.enable_model_cpu_offload,
        "vae_slicing": args.vae_slicing,
        "params_inference": params_inference,
    }

    new_config = OmegaConf.create(new_config)
    return new_config


def get_scheduler(scheduler_name):
    if scheduler_name == "EulerAncestralDiscreteScheduler":
        return EulerAncestralDiscreteScheduler
    elif scheduler_name == "EulerDiscreteScheduler":
        return EulerDiscreteScheduler
    elif scheduler_name == "DEISMultistepScheduler":
        return DEISMultistepScheduler
    elif scheduler_name == "PNDMScheduler":
        return PNDMScheduler
    elif scheduler_name == "KDPM2DiscreteScheduler":
        return KDPM2DiscreteScheduler
    elif scheduler_name == "KDPM2AncestralDiscreteScheduler":
        return KDPM2AncestralDiscreteScheduler
    elif scheduler_name == "DDIMScheduler":
        return DDIMScheduler
    elif scheduler_name == "DDPMScheduler":
        return DDPMScheduler
    elif scheduler_name == "DPMSolverMultistepScheduler":
        return DPMSolverMultistepScheduler
    elif scheduler_name == "HeunDiscreteScheduler":
        return HeunDiscreteScheduler
    elif scheduler_name == "DPMSolverSinglestepScheduler":
        return DPMSolverSinglestepScheduler
    elif scheduler_name == "LMSDiscreteScheduler":
        return LMSDiscreteScheduler
    elif scheduler_name == "UnCLIPScheduler":
        return UnCLIPScheduler
    else:
        raise ValueError(
            f"Scheduler {scheduler_name} not found or implemented in this code"
        )


def get_torch_information():
    if torch.cuda.is_available():
        print("Using GPU")
        nb_gpu = torch.cuda.device_count()
        print(f"Number of GPU: {nb_gpu}")
        print(f"current device {torch.cuda.current_device()}")
        for i in range(nb_gpu):
            print(f"device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Using CPU because no GPU available")


@torch.no_grad()
def prepare_arguments_generation(
    pipeline_type,
    pipeline,
    prompt,
    params_prompt,
    height=None,
    width=None,
    guidance_scale=None,
    num_inference_steps=None,
):
    args_gen = {}
    args_gen["prompt"] = prompt
    if pipeline_type in AVAILABLE_PIPELINE:
        if pipeline_type in [
            "SD",
            "attend_and_excite",
        ]:
            args_gen["height"] = height
            args_gen["width"] = width
            args_gen["guidance_scale"] = guidance_scale
            args_gen["num_inference_steps"] = num_inference_steps
            if pipeline_type == "attend_and_excite":
                args_gen["token_indices"] = []
                for prompt_, params_ in zip(prompt, params_prompt):
                    # detect the indices of the token in the prompt
                    labels_params = params_["labels_params"]
                    to_detect = [
                        pipeline.tokenizer.encode(i, add_special_tokens=False)
                        for i in labels_params.values()
                    ]
                    input_ids = pipeline.tokenizer.encode(
                        prompt_, add_special_tokens=True
                    )
                    token_indices = []
                    for ids_token in to_detect:
                        for i in range(len(input_ids)):
                            if input_ids[i : i + len(ids_token)] == ids_token:
                                token_indices += list(range(i, i + len(ids_token)))
                    args_gen["token_indices"].append(token_indices)
    else:
        raise ValueError(f"pipeline {pipeline_type} not available")
    return args_gen


def diffusers_inference(
    pipeline,
    pipeline_type,
    prompt: List[str],
    params_prompt: List,
    batch_generator,
    iter_per_prompt: int = 1,
    params_inference: dict = {},
):
    images = []
    for i in range(iter_per_prompt):
        bs = len(batch_generator[i]) // len(prompt)
        params_generation = prepare_arguments_generation(
            pipeline_type=pipeline_type,
            pipeline=pipeline,
            prompt=prompt,
            params_prompt=params_prompt,
            **params_inference,
        )
        outputs = pipeline(
            generator=batch_generator[i],
            num_images_per_prompt=bs,
            **params_generation,
        )
        for image in outputs.images:
            images.append(image)
    return images


def check_missing_images(config, prompts):
    with open(config.index, "r") as f:
        already_generated = f.read().splitlines()
    allready_generated = set(already_generated)
    all_prompt = set(prompts)
    missing_prompt = all_prompt - allready_generated
    if len(missing_prompt) > 0:
        print(f"Missing {len(missing_prompt)} prompts")
        print(missing_prompt)
    else:
        print("All prompts generated")


def images_generation(
    batched_prompt,
    batched_params_prompt,
    config,
    seed,
    image_per_prompt,
    device,
    pipeline,
    batch_size,
):
    seed_batch = seed * len(batched_prompt)
    mini_batch_seed = create_packing_list(seed_batch, batch_size)
    iter_per_prompt = math.ceil(image_per_prompt / batch_size)
    batch_generator = []
    for mini_b in mini_batch_seed:
        batch_generator.append(
            [torch.Generator(device=device).manual_seed(s) for s in mini_b]
        )

    return diffusers_inference(
        batch_generator=batch_generator,
        iter_per_prompt=iter_per_prompt,
        prompt=batched_prompt,
        params_prompt=batched_params_prompt,
        pipeline=pipeline,
        pipeline_type=config.pipeline,
        params_inference=config.params_inference,
    )


def generate_description(config):
    return OmegaConf.to_yaml(config)


def create_packing_list(objects: List, n: int):
    """
    Create a list of list of size n from a list of objects
    """
    return [objects[i : i + n] for i in range(0, len(objects), n)]


def write_in_tarfile(path_tar, prompt: List[str], images: Image):
    tar = tarfile.open(path_tar, "a")
    images_per_prompt = len(images) // len(prompt)
    images_pack = create_packing_list(images, images_per_prompt)
    for prompt_, images_ in zip(prompt, images_pack):
        for seed, image in enumerate(images_):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="png")
            img_bytes.seek(0)
            tar_info = tarfile.TarInfo(name=f"{prompt_.replace(' ', '_')}_{seed}.png")
            tar_info.size = len(img_bytes.getvalue())
            tar.addfile(tar_info, img_bytes)
    tar.close()


def write_in_index(path_file, prompt: List[str]):
    with open(path_file, "a") as f:
        for p in prompt:
            f.write(p + "\n")


def set_up_pipeline(pipeline, config, rank):
    device = torch.device(rank)
    if config.pipeline not in ["unclip", "IF"]:
        pipeline.safety_checker = None
        scheduler = get_scheduler(config.scheduler)
        pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)

        if config.enable_model_cpu_offload:
            pipeline.enable_model_cpu_offload(gpu_id=rank)
        else:
            pipeline.to(device)
        if config.vae_slicing:
            pipeline.enable_vae_slicing()
    elif config.pipeline == "IF":
        pipeline.to(rank)
    else:
        pipeline.to(device)


def generation_inference(
    rank,
    queue,
    config,
    lock_tar,
    lock_index,
    prompt_per_batch,
    image_per_prompt,
    batch_size,
):
    """Each subprocess will run this function on a different GPU which is indicated by the parameter `rank`."""
    # get_pipeline
    pipeline = get_pipeline_diffusion(
        pipeline=config.pipeline,
        model_path=config.model_path,
        precision=config.precision,
    )
    set_up_pipeline(pipeline, config, rank)
    seed = list(range(image_per_prompt))
    batched_prompt = []
    batched_params_prompt = []
    generate = False
    last_infer = False
    device = torch.device(rank)
    while True:
        if last_infer:
            break
        prompt, params_prompt = queue.get()
        if prompt is None and params_prompt is None and len(batched_prompt) == 0:
            break
        elif prompt is not None and params_prompt is not None:
            batched_prompt.append(prompt)
            batched_params_prompt.append(params_prompt)
            if len(batched_prompt) == prompt_per_batch:
                generate = True
        elif prompt is None and params_prompt is None and len(batched_prompt) > 0:
            # other case: prompt is None but batch is not empty
            generate = True
            last_infer = True
        if generate:
            # generate images
            images = images_generation(
                image_per_prompt=image_per_prompt,
                batched_prompt=batched_prompt,
                batched_params_prompt=batched_params_prompt,
                config=config,
                device=device,
                pipeline=pipeline,
                seed=seed,
                batch_size=batch_size,
            )

            # exclusif access to a file
            lock_tar.acquire()
            write_in_tarfile(
                path_tar=config.save_dir, images=images, prompt=batched_prompt
            )
            lock_tar.release()
            lock_index.acquire()
            write_in_index(
                path_file=config.index,
                prompt=batched_prompt,
            )
            lock_index.release()
            generate = False
            batched_prompt = []
            batched_params_prompt = []

    print(f"Process {rank} finished")
