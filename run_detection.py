import argparse
import os
import shutil
from pathlib import Path
from typing import List

import torch
from datasets import disable_caching, load_from_disk
from omegaconf import OmegaConf
from torch.multiprocessing import Lock, Pipe, Process, set_start_method

from src.tiam.coco_data import getlabels2numbers
from src.tiam.utils import map_yolo_detection_piped, yolo_inference


def create_packing_list(objects: List, n: int):
    """Create a packing list of n objects."""
    return [objects[i : i + n] for i in range(0, len(objects), n)]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config.yaml file. See example in config/yolo/",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--images_path",
        type=str,
        required=False,
        default=None,
        help="Path to the tarball of images",
    )
    parser.add_argument(
        "--index_yolo",
        type=str,
        required=False,
        default=None,
        help="path to the index file, i.e. list of prompts that have been processed by the detection model. Enable to restart a generation."
        "If not indicated will create `index_yolo.txt` in the same directory as the dataset",
    )
    parser.add_argument("--dataset_path", type=str, required=False, default=None)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--path_results",
        type=str,
        default=None,
        required=False,
        help="Path to save the results (images with boxes). You restart a generation, we will add the new results to the existing one.",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=False,
        default=None,
        help="task to perform, either 'detection' or 'segmentation'. Use the right model for the right task",
    )
    parser.add_argument(
        "--masks_path",
        type=str,
        required=False,
        help="Path to the folder to save the masks (if task is segmentation).",
    )
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--batch_size", type=int, required=False, default=None)

    args = parser.parse_args()
    if args.config is not None:
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({})
    if args.model_path is not None:
        config["model_path"] = args.model_path
    if args.images_path is not None:
        config["images_path"] = args.images_path
    if args.dataset_path is not None:
        config["dataset_path"] = args.dataset_path
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.index_yolo is None and "index_yolo" not in config.keys():
        config["index_yolo"] = args.index_yolo
    if args.path_results is not None:
        config["save_results"] = args.path_results
    if args.overwrite is not None:
        config["overwrite"] = args.overwrite
    if args.task is not None:
        config["task"] = args.task
    if args.masks_path is not None:
        config["masks_path"] = args.masks_path
    if config.task is None:
        config.task = "detection"

    already_processed = []

    if config.index_yolo is not None:
        if os.path.isfile(config.index_yolo):
            if config.overwrite:
                os.remove(config.index_yolo)
            else:
                with open(config.index_yolo, "r") as f:
                    already_processed = f.read().splitlines()
        else:
            with open(config.index_yolo, "w") as f:
                pass
    else:
        if os.path.exists(Path(config.dataset_path).parent / "index_yolo.txt"):
            if config.overwrite:
                os.remove(Path(config.dataset_path).parent / "index_yolo.txt")
            else:
                raise ValueError(
                    "There is already an index file in the save_dir, please use the overwrite flag if you want to replace it"
                    "or change the save_dir argument to resume the generation and add image in the tarfile"
                )
        config.index_yolo = Path(config.dataset_path).parent / "index_yolo.txt"
    already_processed = set(already_processed)

    if config.task == "segmentation":
        if config.masks_path is None:
            raise ValueError("Please provide a path to save the masks")
        else:
            if not os.path.exists(config.masks_path):
                os.makedirs(config.masks_path)
            elif config.overwrite:
                shutil.rmtree(config.masks_path)
                os.makedirs(config.masks_path)

    if config.save_results is not None:
        if config.save_results.endswith(".tar"):
            if config.overwrite and os.path.isfile(config.save_results):
                os.remove(config.save_results)
        elif os.path.isdir(config.save_results):
            config.save_results = os.path.join(config.save_results, "results.tar")
            if os.path.isfile(config.save_results):
                if config.overwrite:
                    os.remove(config.save_results)
                else:
                    raise ValueError(
                        f"A tarball for results already exists in {config.save_results}, please use the overwrite flag if you want to replace it, or precise the path to the tarball with --path_results"
                    )
        print("results will be saved in ", config.save_results)

    params_yolo = {
        "conf": 0.25,
        "iou": 0.8,
        "half": False,
        "show": False,
        "save": False,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
        "show_labels": True,
        "show_conf": True,
        "max_det": 300,
        "vid_stride": False,
        "line_thickness": 2,
        "visualize": False,
        "augment": False,
        "agnostic_nms": False,
        "retina_masks": True,
        "boxes": True,
        "verbose": False,
    }

    disable_caching()
    dataset = load_from_disk(config.dataset_path)
    params_datasets = OmegaConf.create(dataset.info.description)
    images_per_prompt = params_datasets.image_per_prompt

    nb_gpu = torch.cuda.device_count()
    processes = []
    pipes = []
    if nb_gpu >= 1:
        if nb_gpu > 1:
            set_start_method("spawn")
            lock_index = Lock()
            if config.save_results is not None:
                lock_tar = Lock()
        else:
            lock_index = None
            lock_tar = None
        labels2number = getlabels2numbers()
        for rank in range(nb_gpu):
            params_yolo_ = params_yolo.copy()
            params_yolo_["device"] = torch.device(rank)
            pipes.append(Pipe())
            p = Process(
                target=yolo_inference,
                args=(
                    pipes[rank][1],
                    config.model_path,
                    params_yolo_,
                    config.masks_path,
                    config.save_results,
                    config.index_yolo,
                    config.task,
                    config.batch_size,
                    labels2number,
                    config.images_path,
                    images_per_prompt,
                    lock_tar,
                    lock_index,
                    rank,
                ),
            )
            p.start()
            processes.append(p)

    else:
        params_yolo_ = params_yolo.copy()
        params_yolo_["device"] = torch.device("cpu")
        pipes.append(Pipe())
        p = Process(
            target=yolo_inference,
            args=(
                pipes[rank][1],
                config.model_path,
                params_yolo_,
                config.masks_path,
                config.save_results,
                config.index_yolo,
                config.task,
                config.batch_size,
                labels2number,
                config.images_path,
                images_per_prompt,
            ),
        )
        p.start()
        processes.append(p)

    fn_kwargs = {
        "pipes": pipes,
        "already_processed": already_processed,
        "task": config.task,
    }
    if nb_gpu <= 1:
        fn_kwargs["rank"] = 0

    print("Start processing")
    dataset = dataset.map(
        function=map_yolo_detection_piped,
        batched=False,
        fn_kwargs=fn_kwargs,
        with_rank=True if nb_gpu > 1 else False,
        num_proc=nb_gpu if nb_gpu > 1 else None,
    )
    dataset.save_to_disk(config.dataset_path)

    for pipe in pipes:
        pipe[0].send((None, None))
    for p in processes:
        p.join()
    print("Done")


if __name__ == "__main__":
    main()
