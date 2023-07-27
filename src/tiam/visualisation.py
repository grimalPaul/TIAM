import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame


def format_to_pandas(
    dataset, att_binding: bool, conf: List[int], images_per_prompt: int
):
    n_objects = len(dataset["labels_params"][0])
    key_generation_score = "results_for_prompt_conf_"
    key_score = "conf_min_seg"

    key_count_order = "count_order_conf_"
    key_score_seed = "results_conf_"

    key_binding = "conf_min_seg_binding"
    key_pos = "correct_binding_per_pos_of_class_associated_with_color_conf_"
    data = {}
    data_seed = {}
    for c in conf:
        key_generation_conf = key_generation_score + str(c)
        key_count_order_conf = key_count_order + str(c)
        generation_score = dataset[key_generation_conf]
        # wo binding
        data[f"TIAM_wo_binding_{c}"] = [
            generation_score[i][key_score] for i in range(len(generation_score))
        ]

        count = dataset[key_count_order_conf]

        for o in range(n_objects):
            data[f"count_{o}_{c}"] = [count[i][key_score][o] for i in range(len(count))]

        # binding
        if att_binding:
            data[f"TIAM_w_binding_{c}"] = [
                generation_score[i][key_binding] for i in range(len(generation_score))
            ]
            key_pos_conf = key_pos + str(c)
            pos = dataset[key_pos_conf]
            for o in range(n_objects):
                data[f"pos_{o}_{c}"] = [pos[i][key_binding][o] for i in range(len(pos))]
            adjs = dataset["adjs_params"]
            for i in range(len(adjs[0])):
                data[f"adj_{i+1}"] = [a[f"adj{i +1}"] for a in adjs]

        # seed score wo binding
        key_seed_conf = key_score_seed + str(c)
        seeds_scores = {i: [] for i in range(images_per_prompt)}
        data_score = dataset[key_seed_conf]
        for score in data_score:
            for i, s in enumerate(score):
                seeds_scores[i].append(s[key_score][0])

        df_seed = DataFrame(data=seeds_scores)
        data_seed[c] = df_seed.to_numpy().mean(axis=0)
        if att_binding:
            for o in range(n_objects):
                divisor = np.array(data[f"count_{o}_{c}"]) * images_per_prompt
                args = np.argwhere(divisor == 0).flatten()
                divisor[args] = 1
                data[f"prop_{o}_{c}"] = np.array(data[f"pos_{o}_{c}"]) / divisor
                data[f"prop_{o}_{c}"][args] = np.nan

    return DataFrame(data=data), DataFrame(data=data_seed), n_objects


def generate_table(
    df: DataFrame,
    conf: List[int],
    att_binding: bool,
    images_per_prompt: int = None,
    path_to_save: str = None,
):
    n_prompts = df.shape[0]
    columns = [f"TIAM_wo_binding_{c}" for c in conf]
    if att_binding:
        columns += [f"TIAM_w_binding_{c}" for c in conf]
    desc_conf = "Confidence threshold"
    desc_wo_binding = (
        "TIAM score with object ground truth only" if att_binding else "TIAM score"
    )
    desc_binding = "TIAM score with object and attribute ground truth"
    score = df[columns].mean().reset_index()
    score[desc_conf] = score["index"].apply(lambda x: x.split("_")[-1])
    score["binding"] = score["index"].apply(
        lambda x: desc_wo_binding if x.split("_")[1] == "wo" else desc_binding
    )
    score = score.drop(columns=["index"])
    score = score.pivot(index=desc_conf, columns="binding", values=0)
    score.reset_index(inplace=True)
    score.columns.name = None

    infos = (
        f" Compute for {n_prompts} prompts with {images_per_prompt} images per prompt."
    )

    score = score.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    score = score.set_index(desc_conf)
    score.style.to_latex(
        os.path.join(path_to_save, "TIAM_score.tex"),
        caption="TIAM score with and without attribute ground truth." + infos
        if att_binding
        else "TIAM score." + infos,
    )
    print("TIAM score.", infos)
    print(score.to_markdown(tablefmt="github"))


def objects_occurence_by_position(
    df: DataFrame, conf: List[int], n_objects: int, path_to_save: str
):
    columns = [f"count_{o}_{c}" for o in range(n_objects) for c in conf]
    score = df[columns].mean().reset_index()
    desc_conf = "Confidence threshold"
    score["position"] = score["index"].apply(lambda x: x.split("_")[1])
    score[desc_conf] = score["index"].apply(lambda x: x.split("_")[-1])
    score["position"] = score["position"].astype(int).apply(lambda x: f"$o_{x + 1}$")

    fig, ax = plt.subplots(figsize=(2 * len(conf), 5))
    sns.barplot(data=score, x=desc_conf, y=0, hue="position", ax=ax)
    ax.set_ylabel("Proportion of objects in images")
    ax.legend(loc="lower right", fontsize=11, ncol=4)
    ax.grid(axis="y", which="major", linestyle="--", linewidth="0.5", color="gray")
    ax.set_title(
        f"The proportion of occurrences of each object, based on its positions in the prompt.\nThe template of the prompt has {n_objects} objects."
    )
    fig.savefig(
        os.path.join(path_to_save, "proportion_of_occurence_of_objects.png"),
        dpi=300,
        bbox_inches="tight",
    )


def boxplot_TIAM_seeds(df: DataFrame, path_to_save: str, n_prompts: int):
    # only considering the TIAM score of object ground truth (without binding)
    fig, ax = plt.subplots(figsize=(2 * df.shape[1], 5))
    sns.boxplot(
        data=df,
        color="skyblue",
        ax=ax,
        showfliers=True,
        fliersize=3,
        linewidth=1,
        showmeans=True,
        meanprops={
            "marker": "+",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
        },
    )
    ax.set_ylabel("TIAM")
    ax.set_xlabel("Confidence threshold")
    ax.grid(axis="y", alpha=0.5)
    infos = f"{df.shape[0]} seeds were used to generate images.\nScores are average on the {n_prompts} prompts."
    ax.set_title(
        "Boxplot of the TIAM score per seed according to the confidence threshold.\n"
        + infos
    )
    fig.savefig(
        os.path.join(path_to_save, "boxplot_seed_TIAM.png"),
        dpi=300,
        bbox_inches="tight",
    )


def success_rate_attribute_binding(
    df: DataFrame, conf: List[int], path_to_save: str, n_objects: int
):
    columns = [f"prop_{o}_{c}" for o in range(n_objects) for c in conf]
    score = df[columns].mean().reset_index()
    # split columns to position and conf
    desc_conf = "Confidence threshold"
    score["position"] = score["index"].apply(lambda x: x.split("_")[1])
    score[desc_conf] = score["index"].apply(lambda x: x.split("_")[-1])
    score["position"] = score["position"].astype(int).apply(lambda x: f"$o_{x + 1}$")

    # remove row with nan values
    score = score.dropna()
    fig, ax = plt.subplots(figsize=(2 * len(conf), 5))
    sns.barplot(data=score, x=desc_conf, y=0, hue="position", ax=ax)

    ax.set_ylabel("Binding success rate among detected object")
    ax.legend(loc="lower right", fontsize=11, ncol=4)
    ax.grid(axis="y", which="major", linestyle="--", linewidth="0.5", color="gray")
    ax.set_title(
        f"Binding success rate of color attribution w.r.t the detected objects."
    )
    fig.savefig(
        os.path.join(path_to_save, "binding_success_rate.png"),
        dpi=300,
        bbox_inches="tight",
    )


def binding_success_rate(
    df: DataFrame, conf: List[int], path_to_save: str, n_objects: int
):
    colors_palette = {
        "blue": "#1f77b4",
        "orange": "#ff7f0e",
        "green": "#2ca02c",
        "red": "#ff0000",
        "purple": "#9467bd",
        "pink": "#e377c2",
        "yellow": "#ffcd35",
    }
    desc_conf = "Confidence threshold"
    for o in range(n_objects):
        columns = [f"prop_{o}_{c}" for c in conf] + [f"adj_{o+1}"]
        score = df[columns].groupby([f"adj_{o+1}"]).mean().reset_index()
        x_name = f"Binding success rate $a_{o+1}$"
        score = score.melt(
            id_vars=[f"adj_{o+1}"], var_name=desc_conf, value_name=x_name
        )
        score[desc_conf] = score[desc_conf].apply(lambda x: x.split("_")[-1])
        score = score.dropna()
        fig, ax = plt.subplots(figsize=(2 * len(conf), 5))
        sns.barplot(
            data=score,
            x=desc_conf,
            y=x_name,
            hue=f"adj_{o+1}",
            ax=ax,
            palette=colors_palette,
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1)
        ax.grid(axis="y", linestyle="-", alpha=0.5)
        ax.set_title(
            f"Binding success rate for the $o_{o + 1}$ object, among the $o_{o + 1}$ objects correctly detected.",
        )
        fig.savefig(
            os.path.join(path_to_save, f"binding_success_rate_attribute_{o}.png"),
            dpi=300,
            bbox_inches="tight",
        )
