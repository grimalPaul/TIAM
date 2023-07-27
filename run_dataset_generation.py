import argparse

from omegaconf import OmegaConf

from src.prompts_dataset.factory import multi_combinations, unique_words
from src.prompts_dataset.utils import (
    generate_dataset,
    get_generate_fct,
    get_template_params,
    get_words,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to the directory where the dataset will be saved.",
    )

    parser.add_argument(
        "--file_name",
        type=str,
        help="Name of the folder where the generated data will be saved. Just named without extension if you want to generate dataset (from datasets library). If you end the text by .txt, it will generate a sentence per line in a file.txt",
        default="dataset",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config.yaml file. See example in config/",
    )

    parser.add_argument(
        "--txt",
        action="store_true",
        help="Will generate a text file. If not specified, will generate a dataset from datasets library",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    template, params = get_template_params(config)

    # filter a/an word
    apply_on = []  # (indefinite_article, word)
    params_with_possible_repetition = template.get_all_identifiers()
    for i, p in enumerate(params_with_possible_repetition):
        if p.startswith("ind_ar"):
            # search for the corresponding word and search the next param
            apply_on.append((p, params_with_possible_repetition[i + 1]))
            params.remove(p)
    # get adj and label
    labels = []
    ajds = []
    adj_label = {}
    for i, p in enumerate(params):
        if p.startswith("object"):
            labels.append(p)
        if p.startswith("adj"):
            label = "object" + p.split("adj")[1]
            adj_label[p] = label
            ajds.append(p)
    print(
        f"We detected the following labels: {labels}\
          \nand the following adjectives: {ajds}\
          \nBe sure that adj and label correspond {adj_label}"
    )

    multi_list = True
    for param in params:
        if param not in config.keys():
            multi_list = False

    if not multi_list:
        if "objects" in config.keys():
            words = get_words(config.objects)
        else:
            raise ValueError(
                "You should specify a list of 'objects' in config.\
                see example in config/coco_config.yaml \n\
                You can also specify some list of objects according to your template.\
                See an example in config/prompts_datasets"
            )
        n = len(params)
        generate = get_generate_fct(config.generate)
        iter_words = generate(words, n)

    else:
        # multi_list
        words = []
        for param in params:
            words.append(get_words(config[param]))
        iter_words = multi_combinations(words)
        if "unique" in config.keys():
            unique = OmegaConf.to_object(config.unique)
            if len(unique) != 0:
                positions = []
                for not_be_repeated in unique:
                    positions.append([params.index(p) for p in not_be_repeated])
                iter_words = unique_words(iter_words, positions)

    generate_dataset(
        iter_words=iter_words,
        template=template,
        directory=args.save_dir,
        dataset=False if args.txt else True,
        file_name=args.file_name,
        labels_params=labels,
        adj_params=ajds,
        adj_apply_on=adj_label,
        template_params=params,
        indefinite_article_params=apply_on,
    )


if __name__ == "__main__":
    main()
