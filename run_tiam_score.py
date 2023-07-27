import argparse
import os
import shutil

from datasets import disable_caching, load_from_disk
from omegaconf import OmegaConf
from torch.multiprocessing import Manager

from src.tiam.utils import create_params_compute_stats, map_compute_stats
from src.tiam.visualisation import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default=None,
        help="path to the dataset of prompts",
    )

    parser.add_argument(
        "--eval_precomputed",
        action="store_true",
        help="if specify, we will not compute the TIAM score but only the visualisation",
    )

    parser.add_argument(
        "--path_to_masks",
        type=str,
        required=False,
        default=None,
        help="root where the folder containing the masks are saved, we will use that to load masks.",
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="number of process to use for multiprocessing",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        default=None,
        help="path to save TIAM results and json files with classification",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="save json files with classification of the images (success/fail)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite the results in the results_json and/or in results folder if they already exist",
    )

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.save_dir is not None:
        config["save_dir"] = args.save_dir
    if args.dataset_path is not None:
        config["dataset_path"] = args.dataset_path
    if args.path_to_masks is not None:
        config["path_to_masks"] = args.path_to_masks
    if args.num_proc is not None:
        lock = Manager().Lock()
    else:
        lock = None
    config["save_dir"] = args.save_dir
    config["overwrite"] = args.overwrite
    config["json"] = args.json
    disable_caching()

    dataset = load_from_disk(config.dataset_path)

    if "adjs_params" in dataset.column_names and len(dataset["adjs_params"][0]) > 0:
        print("TIAM score with attribute binding will also be computed")
        config["binding"] = True
    else:
        config["binding"] = False
    conf = OmegaConf.to_object(config.params_compute_stats.conf)

    params_datasets = OmegaConf.create(dataset.info.description)
    images_per_prompt = params_datasets.image_per_prompt

    if not args.eval_precomputed:
        # we cannot use batched=True because we iterate per prompt
        if not isinstance(conf, list):
            conf = [conf]
        params_compute_stats = OmegaConf.to_object(config.params_compute_stats)
        params_compute_stats.pop("conf")
        params = create_params_compute_stats(**params_compute_stats)
        params["image_size"] = (
            (512, 512) if "image_size" not in config else config.image_size
        )
        params["path_to_masks"] = config.path_to_masks
        params["images_per_prompt"] = images_per_prompt
        params["lock_json"] = lock
        params["binding"] = config.binding
        for c in conf:
            dataset = load_from_disk(config.dataset_path)
            dataset.cleanup_cache_files()
            params["conf_min"] = c
            params["name_new_column"] = f"conf_{c}"

            params["path_to_save_json"] = (
                os.path.join(config.save_dir, "results_json", f"conf_{c}")
                if config.json
                else None
            )
            if config.json:
                if os.path.exists(params["path_to_save_json"]):
                    if config.overwrite:
                        shutil.rmtree(params["path_to_save_json"])
                    else:
                        raise ValueError(
                            "json folder already exists. Precise --overwrite to overwrite it."
                        )

            dataset = dataset.map(
                function=map_compute_stats,
                batched=False,
                fn_kwargs=params,
                num_proc=args.num_proc,
                keep_in_memory=True,
            )
            print(f"saving dataset for {c}")
            dataset.save_to_disk(config.dataset_path)
            del dataset
    dataset = load_from_disk(config.dataset_path)

    path_to_save = os.path.join(config.save_dir, "TIAM_score")
    if os.path.exists(path_to_save):
        if config.overwrite:
            shutil.rmtree(path_to_save)
        else:
            raise ValueError(
                "results folder already exists. Precise --overwrite to overwrite it."
            )
    os.makedirs(path_to_save, exist_ok=True)

    df, seeds, n_objects = format_to_pandas(
        dataset=dataset,
        att_binding=config.binding,
        conf=conf,
        images_per_prompt=images_per_prompt,
    )
    generate_table(
        df,
        att_binding=config.binding,
        conf=conf,
        images_per_prompt=images_per_prompt,
        path_to_save=path_to_save,
    )
    objects_occurence_by_position(
        df,
        conf=conf,
        n_objects=n_objects,
        path_to_save=path_to_save,
    )
    boxplot_TIAM_seeds(
        seeds,
        n_prompts=dataset.num_rows,
        path_to_save=path_to_save,
    )
    if config.binding:
        success_rate_attribute_binding(
            df,
            conf=conf,
            path_to_save=path_to_save,
            n_objects=n_objects,
        )
        binding_success_rate(
            df,
            conf=conf,
            path_to_save=path_to_save,
            n_objects=n_objects,
        )


if __name__ == "__main__":
    main()
