import os
from string import Template

from datasets import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from .factory import all_combinations, combinations, get_article, permutation


class Template_(Template):
    def get_all_identifiers(self):
        ids = []
        for mo in self.pattern.finditer(self.template):
            named = mo.group("named") or mo.group("braced")
            if named is not None:
                # add a named group
                ids.append(named)
            elif (
                named is None
                and mo.group("invalid") is None
                and mo.group("escaped") is None
            ):
                # If all the groups are None, there must be
                # another group we're not expecting
                raise ValueError("Unrecognized named group in pattern", self.pattern)
        return ids


def get_template_params(config):
    template = Template_(config.template)
    params = template.get_identifiers()
    return template, params


def get_generate_fct(generate: str):
    if generate == "permutation":
        return permutation
    elif generate == "combination":
        return all_combinations
    elif generate == "ordered_combination":
        return combinations
    else:
        raise ValueError(
            "generate should be in ['permutation','combination','ordered_combination']\
            See in README.md for more details"
        )


def get_words(config):
    if "list" in config.keys() and config.list is not None:
        return OmegaConf.to_object(config.list)
    elif "path" in config.keys() and config.path is not None:
        return load_words(config.path)
    else:
        raise ValueError(
            "You should specify a list of 'words' in config. \
            Or a path to a file containing the list of words. \
            See README.md for more details."
        )


def load_words(path):
    # get the relativ path
    with open(path, "r") as f:
        words = [line.rstrip() for line in f]
    return words


def generate_dataset(
    iter_words,
    template,
    directory,
    template_params,
    adj_params=None,
    labels_params=None,
    adj_apply_on=None,
    indefinite_article_params=None,
    file_name: str = "dataset",
    dataset: bool = False,
):
    sentences = []
    # all params
    params = []
    # only labels that we will detect
    labels = []
    # only adj that we will check
    adjs = []
    # wich adj is applied on which label
    for words in tqdm(iter_words):
        param = dict(zip(template_params, words))
        params.append(param.copy())

        adjs_ = {adj: param[adj] for adj in adj_params}
        labels_ = {label: param[label] for label in labels_params}
        adjs.append(adjs_)
        labels.append(labels_)
        if indefinite_article_params:
            for indefinite_article, word in indefinite_article_params:
                param[indefinite_article] = get_article(param[word])
        sentence = template.substitute(param)
        sentences.append(sentence)
    if dataset and not file_name.endswith(".txt"):
        # remove indefinite article from params
        dataset = Dataset.from_dict(
            {
                "sentence": sentences,
                "template": [template.template] * len(sentences),
                "params": params,
                "labels_params": labels,
                "adjs_params": adjs,
                "adj_apply_on": [adj_apply_on] * len(sentences),
            }
        )
        dataset.save_to_disk(os.path.join(directory, file_name))
    else:
        if not file_name.endswith(".txt"):
            file_name += ".txt"
        with open(os.path.join(directory, file_name), "w") as f:
            for i, sentence in enumerate(sentences):
                if i != 0:
                    f.write("\n")
                f.write(sentence)
