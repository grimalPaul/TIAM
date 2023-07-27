from argparse import ArgumentParser

from datasets import disable_caching, load_from_disk
from omegaconf import OmegaConf

from src.tiam.utils import map_attribute_binding, open_tarball

disable_caching()


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--images_path", type=str, help="path to the tarball of images")
    parser.add_argument("--config", type=str, help="config file for the experiment")
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
    args = parser.parse_args()
    if args.config is not None:
        config = OmegaConf.load(args.config)
        if args.dataset_path is not None:
            config.dataset_path = args.dataset_path
        if args.images_path is not None:
            config.images_path = args.images_path
        if args.path_to_masks is not None:
            config.path_to_masks = args.path_to_masks
        if args.num_proc is not None:
            config.num_proc = args.num_proc
    else:
        config = args

    tarball_images = open_tarball(config.images_path)

    fn_kwargs = {
        "path_to_masks": config.path_to_masks,
        "tar_images": tarball_images,
    }

    dataset = load_from_disk(config.dataset_path)
    params_datasets = OmegaConf.create(dataset.info.description)
    fn_kwargs["images_per_prompt"] = params_datasets.image_per_prompt
    dataset = dataset.map(
        map_attribute_binding,
        batched=False,
        input_columns=["sentence"],
        fn_kwargs=fn_kwargs,
        num_proc=config.num_proc,
    )
    dataset.save_to_disk(config.dataset_path)


if __name__ == "__main__":
    main()
