import os
import shutil
import tarfile
import time
from pathlib import Path
from queue import Full

import torch
from datasets import disable_caching, load_from_disk
from omegaconf import OmegaConf
from torch.multiprocessing import Lock, Process, Queue, set_start_method
from tqdm import tqdm

from src.diffusion.utils import (
    check_missing_images,
    create_config,
    generate_description,
    generation_inference,
    get_pipeline_diffusion,
    get_scheduler,
    get_torch_information,
    images_generation,
    parse_args,
    write_in_index,
    write_in_tarfile,
)


def main():
    args = parse_args()
    config_yaml = OmegaConf.load(args.config)
    config = create_config(args, config_yaml)

    batch_size = config.batch_size
    image_per_prompt = config.image_per_prompt

    # we iter one by one on the prompt
    if int(batch_size / image_per_prompt) == 1:
        batch_size = image_per_prompt

    # if batch size > image per prompt and batch size is a multiple of image per prompt
    # we can do more than one prompt per batch
    if batch_size > image_per_prompt:
        if batch_size / image_per_prompt >= 2:
            # can do more than one prompt per batch
            prompt_per_batch = batch_size // image_per_prompt
            batch_size = image_per_prompt * prompt_per_batch
        else:
            batch_size = image_per_prompt
    else:
        prompt_per_batch = 1

    disable_caching()
    dataset = load_from_disk(config.dataset_path)

    get_torch_information()
    if not config.save_dir.endswith(".tar"):
        # check if the save_dir exist
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        else:
            # check if the tarfile images.tar already exist in the save_dir
            if os.path.exists(os.path.join(config.save_dir, "images.tar")):
                if config.overwrite:
                    # delete the tarfile
                    os.remove(os.path.join(config.save_dir, "images.tar"))
                else:
                    raise ValueError(
                        f"There is already a tarfile in the save_dir, please use the overwrite flag if you want to replace it"
                        f"or change the save_dir argument to resume the generation and add image in the tarfile {os.path.join(config.save_dir, 'images.tar')}"
                    )
        config.save_dir = os.path.join(config.save_dir, "images.tar")
        tar = tarfile.open(config.save_dir, "w")
        tar.close()
    else:
        # check if the tarfile exist
        if not os.path.exists(config.save_dir):
            os.makedirs(Path(config.save_dir).parent)
            tar = tarfile.open(config.save_dir, "w")
            tar.close()

    already_generated = []
    if config.index is not None:
        # check if the index exist
        if os.path.isfile(config.index):
            if config.overwrite:
                # delete the index
                os.remove(config.index)
            else:
                with open(config.index, "r") as f:
                    already_generated = f.read().splitlines()
        else:
            with open(config.index, "w") as f:
                pass
    else:
        # check if the index exist
        if os.path.exists(Path(config.save_dir).parent / "index.txt"):
            if config.overwrite and os.path.isfile(
                Path(config.save_dir).parent / "index.txt"
            ):
                # delete the index
                os.remove(Path(config.save_dir).parent / "index.txt")
            else:
                raise ValueError(
                    "There is already an index file in the save_dir, please use the --overwrite flag if you want to replace it."
                    "Or change the --index_file argument to resume the generation and add image in the tarfile"
                )
        config.index = Path(config.save_dir).parent / "index.txt"

    if os.path.exists(os.path.join(Path(config.save_dir).parent, "dataset_prompt")):
        if config.overwrite:
            # delete the index
            shutil.rmtree(os.path.join(Path(config.save_dir).parent, "dataset_prompt"))
        else:
            raise ValueError(
                "There is already a dataset_prompt folder in the save_dir, please use the overwrite flag if you want to replace it"
            )
    already_generated = set(already_generated)

    print(f"Pipeline used {config.pipeline}, scheduler {config.scheduler}")
    print(
        f"precision {config.precision}, batch_size {config.batch_size}, image_per_prompt {config.image_per_prompt}"
    )
    print(f"config.params_inference {config.params_inference}")
    print(
        f"Images will be saved in {config.save_dir}, Index will be saved in {config.index}"
    )

    # multiple gpu or one gpu
    nb_gpu = torch.cuda.device_count()
    if nb_gpu > 1:
        set_start_method("spawn")
        # multiprocessing
        queue = Queue(maxsize=nb_gpu)
        lock_tar = Lock()
        lock_index = Lock()
        processes = []
        for rank in range(nb_gpu):
            p = Process(
                target=generation_inference,
                args=(
                    rank,
                    queue,
                    config,
                    lock_tar,
                    lock_index,
                    prompt_per_batch,
                    image_per_prompt,
                    batch_size,
                ),
            )
            p.start()
            processes.append(p)

        for batch_prompt in tqdm(
            dataset.select_columns(
                ["sentence", "labels_params", "adjs_params", "adj_apply_on"]
            ),
            desc="Generating images",
        ):
            # check if the prompt is already generated
            prompt = batch_prompt["sentence"]
            params_prompt = {
                "labels_params": batch_prompt["labels_params"],
                "adjs_params": batch_prompt["adjs_params"],
                "adj_apply_on": batch_prompt["adj_apply_on"],
            }
            if prompt not in already_generated:
                # block until a free slot is available
                try:
                    queue.put((prompt, params_prompt), timeout=600)
                except Full:
                    # check if process are still alive
                    terminate = False
                    for p in processes:
                        if not p.is_alive():
                            terminate = True
                    if terminate:
                        for p in processes:
                            p.terminate()
                        # check generated images
                        check_missing_images(config, dataset["sentence"])
                        raise ValueError(
                            "One of the process is not alive anymore, you can resume the generation. Please precise --index_file and --save_dir /file_name.tar"
                        )
                    else:
                        # wait for a free slot
                        while queue.full():
                            time.sleep(2)
                        queue.put((prompt, params_prompt))

        for _ in range(nb_gpu):
            queue.put((None, None))

        dataset.info.description = generate_description(config)
        dataset.save_to_disk(
            os.path.join(Path(config.save_dir).parent, "dataset_prompt")
        )
        # wait the processes
        for p in processes:
            p.join()
        print("All processes finished")

        check_missing_images(config, dataset["sentence"])

    else:
        pipeline = get_pipeline_diffusion(
            pipeline=config.pipeline,
            model_path=config.model_path,
            precision=config.precision,
        )
        if config.pipeline not in ["unclip", "IF"]:
            pipeline.safety_checker = None
            scheduler = get_scheduler(config.scheduler)
            pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)

            if args.attention_slice is not None:
                pipeline.enable_attention_slicing(args.attention_slice)
            if args.mps:
                device = "mps"
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if config.enable_model_cpu_offload:
                pipeline.enable_model_cpu_offload(gpu_id=0)
            else:
                pipeline.to(device)

            if config.vae_slicing:
                pipeline.enable_vae_slicing()
        elif config.pipeline == "IF":
            rank = torch.cuda.current_device()
            pipeline.to(rank)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline = pipeline.to(device)

        seed = list(range(image_per_prompt))

        batched_prompt = []
        batched_params_prompt = []
        num_prompts = dataset.num_rows
        for i, batch_prompt in enumerate(
            tqdm(
                dataset.select_columns(
                    ["sentence", "labels_params", "adjs_params", "adj_apply_on"]
                ),
                desc="Generating images",
            )
        ):
            prompt = batch_prompt["sentence"]
            params_prompt = {
                "labels_params": batch_prompt["labels_params"],
                "adjs_params": batch_prompt["adjs_params"],
                "adj_apply_on": batch_prompt["adj_apply_on"],
            }
            if prompt not in already_generated:
                batched_prompt.append(prompt)
                # used for A&E
                batched_params_prompt.append(params_prompt)
            if len(batched_prompt) == prompt_per_batch or i == num_prompts - 1:
                images = images_generation(
                    batch_size=batch_size,
                    batched_prompt=batched_prompt,
                    batched_params_prompt=batched_params_prompt,
                    config=config,
                    device=device,
                    image_per_prompt=image_per_prompt,
                    pipeline=pipeline,
                    seed=seed,
                )

                # save the images
                write_in_tarfile(
                    images=images, path_tar=config.save_dir, prompt=batched_prompt
                )
                write_in_index(path_file=config.index, prompt=batched_prompt)
                batched_prompt = []
                batched_params_prompt = []
        dataset.info.description = generate_description(config)
        # save dataset in the save dir
        dataset.save_to_disk(
            os.path.join(Path(config.save_dir).parent, "dataset_prompt")
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Generation time {end_time - start_time} seconds")
    print(f"Generation time {(end_time - start_time) / 60} minutes")
    print(f"Generation time {(end_time - start_time) / 3600} hours")
