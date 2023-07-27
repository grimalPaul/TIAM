import io
import json
import os
import tarfile
from typing import List

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.v8.segment.predict import SegmentationPredictor

from .attribute_binding import eval_distance_to_colors
from .eval import eval

RESULT_KEYS = [
    "without_filter",
    "with_conf_min",
    "conf_min_bbox",
    "conf_min_seg",
]

COUNT_KEYS = [
    "without_filter",
    "with_conf_min",
    "conf_min_bbox",
    "conf_min_seg",
]

KEYS_BINDING = [
    "without_filter_binding",
    "with_conf_min_binding",
    "conf_min_seg_binding",
]


def get_predictor(task: str):
    if task == "detection":
        return DetectionPredictor()
    elif task == "segmentation":
        return SegmentationPredictor()
    else:
        raise ValueError(
            f"task {task} not recognized. Should be either 'detection' or 'segmentation'."
        )


def create_packing_list(objects: List, n: int):
    """Create a packing list of n objects."""
    return [objects[i : i + n] for i in range(0, len(objects), n)]


def map_yolo_detection_piped(batch, rank, pipes, already_processed, task="detect"):
    prompt = batch["sentence"]
    params = batch["labels_params"]
    if prompt in already_processed:
        return None
    else:
        pipes[rank][0].send((prompt, params))
        bbox_batch, classes_batch, conf_batch = pipes[rank][0].recv()
        return (
            {"bbox": bbox_batch, "classes": classes_batch, "conf": conf_batch}
            if task == "detect"
            else {
                "bbox": bbox_batch,
                "classes": classes_batch,
                "conf": conf_batch,
            }
        )


def yolo_inference(
    pipe,
    model_path,
    params_yolo,
    masks_path,
    results_path_tar,
    index_yolo,
    task,
    batch_size,
    labels2number,
    images_path,
    images_per_prompt,
    lock_tar=None,
    lock_index=None,
    rank=0,
):
    print(f"Starting process {rank}")
    yolo_model = YOLO(model_path)
    yolo_model.predictor = get_predictor(task)

    tarball_images = open_tarball(images_path)

    while True:
        prompt, params = pipe.recv()
        if prompt is None:
            break
        images = []
        for seed in range(0, images_per_prompt):
            # load all the images
            f = tarball_images.extractfile(prompt.replace(" ", "_") + f"_{seed}.png")
            image = Image.open(io.BytesIO(f.read()))
            images.append(image)

        classes2detect = [labels2number[p] for p in params.values()]
        bbox_batch = []
        classes_batch = []
        conf_batch = []

        images = create_packing_list(images, batch_size)

        if task == "segmentation":
            seg_masks = {}
        if results_path_tar is not None:
            results_img = []

        cpt = 0
        for batch_img in images:
            results = yolo_model(
                source=batch_img, classes=classes2detect, **params_yolo
            )
            for seed, r in enumerate(results):
                if task == "segmentation":
                    # if no object detected, masks is None
                    if r.masks is not None:
                        seg_masks.update(
                            {
                                f"{prompt.replace(' ', '_')}_mask_{seed + cpt}_{idx}": m
                                for idx, m in enumerate(r.masks.xy)
                            }
                        )
                if results_path_tar is not None:
                    # return np array of the annotated image
                    results_img.append(
                        Image.fromarray(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
                    )
                r = r.cpu().numpy()
                bbox_batch.append(r.boxes.xyxyn.tolist())
                classes_batch.append(r.boxes.cls.astype(int).tolist())
                conf_batch.append(r.boxes.conf.tolist())
            cpt += len(batch_img)
        pipe.send((bbox_batch, classes_batch, conf_batch))

        # save the masks, results, index
        if task == "segmentation":
            if masks_path is not None:
                file_name = os.path.join(masks_path, f"{prompt.replace(' ', '_')}.npz")
                np.savez_compressed(file_name, **seg_masks)

        if results_path_tar is not None:
            if lock_tar is not None:
                lock_tar.acquire()
            tar = tarfile.open(results_path_tar, "a")
            for seed, res_plotted in zip(range(0, images_per_prompt), results_img):
                img_bytes = io.BytesIO()
                res_plotted.save(img_bytes, format="png")
                img_bytes.seek(0)
                tar_info = tarfile.TarInfo(
                    name=f"{prompt.replace(' ', '_')}_{seed}.png"
                )
                tar_info.size = len(img_bytes.getvalue())
                tar.addfile(tar_info, img_bytes)
            tar.close()
            if lock_tar is not None:
                lock_tar.release()

        if lock_index is not None:
            lock_index.acquire()
        with open(index_yolo, "a") as f:
            f.write(prompt + "\n")
        if lock_index is not None:
            lock_index.release()


def save2json(sorted_images, path2save, keys):
    os.makedirs(path2save, exist_ok=True)
    # open and load json file
    # if file exists load it
    if os.path.exists(os.path.join(path2save, "sorted_images.json")):
        with open(os.path.join(path2save, "sorted_images.json"), "r") as f:
            file = json.load(f)
        # update file
        for key in keys:
            sorted_images[key]["correct"] = (
                sorted_images[key]["correct"] + file[key]["correct"]
            )
            sorted_images[key]["failure"] = (
                sorted_images[key]["failure"] + file[key]["failure"]
            )

    with open(os.path.join(path2save, "sorted_images.json"), "w") as f:
        json.dump(sorted_images, f, indent=4)


def create_params_compute_stats(
    conf_min=0.25,
    remove_superposed=False,
    remove_supervised_type=None,
    remove_superposed_iou_min=0.95,
    keep_superposed_with_best_conf=False,
):
    return {
        "conf_min": conf_min,
        "remove_superposed": remove_superposed,
        "remove_supervised_type": remove_supervised_type,
        "remove_superposed_iou_min": remove_superposed_iou_min,
        "keep_superposed_with_best_conf": keep_superposed_with_best_conf,
    }


def map_compute_stats(
    batch,
    name_new_column,
    path_to_masks,
    images_per_prompt,
    binding=False,
    conf_min=0.5,
    remove_superposed=False,
    remove_supervised_type=None,
    remove_superposed_iou_min=0.95,
    keep_superposed_with_best_conf=False,
    path_to_save_json=None,
    image_size=(512, 512),
    lock_json=None,
):
    """
    one batch per prompt, multiple images per prompt
    """

    classes_from_prompt = batch["labels_params"]
    prompt = batch["sentence"]

    if binding:
        # create dict class : color
        colors_from_prompt = {}
        for adj, word in batch["adj_apply_on"].items():
            colors_from_prompt[batch["labels_params"][word]] = batch["adjs_params"][adj]

    results_2_save = []
    nb_items = images_per_prompt
    # result general for the batched prompt
    results_for_prompt = {}
    for key in RESULT_KEYS:
        results_for_prompt[key] = 0

    count_order = {}
    for key in COUNT_KEYS:
        count_order[key] = [0] * len(classes_from_prompt.values())

    if binding:
        correct_binding_per_pos_of_class_associated_with_color = {}
        for key in KEYS_BINDING:
            results_for_prompt[key] = 0
            # [total prompt with all elements nb_A nb_B ...]
            count_order[key] = [0] * (len(colors_from_prompt.values()) + 1)
            # cls detected and bind
            # [nb_A, nb_B, ...] for all the cls A, B, ... cls associated with color in the prompt
            correct_binding_per_pos_of_class_associated_with_color[key] = [0] * (
                len(colors_from_prompt.values())
            )
    if path_to_save_json is not None:
        sorted_images = {}
        for key in RESULT_KEYS:
            sorted_images[key] = {"correct": [], "failure": []}
        if binding:
            for key in KEYS_BINDING:
                sorted_images[key] = {"correct": [], "failure": []}

    if path_to_masks is not None:
        masks = np.load(f"{path_to_masks}/{prompt.replace(' ', '_')}.npz")
        available_masks, prompt2masks = get_available_masks(masks)
    else:
        masks = None
    # iter image per image for one prompt
    for seed, bbox, classes, conf in zip(
        range(images_per_prompt),
        batch["bbox"],
        batch["classes"],
        batch["conf"],
    ):
        if masks is not None:
            masks_name = get_masks_name(available_masks, prompt2masks, prompt, seed)
        else:
            masks_name = None

        if binding:
            binding_args = {
                "detected_colors": batch["labels_distances"][seed],
                "percentage_colors": batch["percentages_distances"][seed],
                "colors_from_prompt": colors_from_prompt,
            }
        else:
            binding_args = {}

        results, count_order_temp = eval(
            classes_from_prompt=classes_from_prompt,
            detected_classes=classes,
            conf=conf,
            remove_superposed_type=remove_supervised_type,
            keep_superposed_with_best_conf=keep_superposed_with_best_conf,
            conf_min=conf_min,
            bbox=bbox,
            masks=masks,
            masks_name=masks_name,
            remove_superposed=remove_superposed,
            remove_superposed_iou_min=remove_superposed_iou_min,
            mask_size=image_size,
            binding=binding,
            **binding_args,
        )

        for key in RESULT_KEYS:
            if key in results:
                results_for_prompt[key] += results[key][0]
            else:
                results_for_prompt[key] = None

        for key in COUNT_KEYS:
            if key in count_order_temp:
                for i in range(len(count_order_temp[key])):
                    count_order[key][i] += count_order_temp[key][i]

        if binding:
            for key in KEYS_BINDING:
                if key in results:
                    results_for_prompt[key] += results[key]
                    cpt_1 = 0  # correct bind
                    cpt_0 = 0  # fail bind
                    cpt_None = 0  # not detected
                    for i, v in enumerate(count_order_temp[key]):
                        if v == 1:
                            cpt_1 += 1
                            correct_binding_per_pos_of_class_associated_with_color[key][
                                i
                            ] += 1
                        elif v == 0:
                            cpt_0 += 1
                        else:
                            cpt_None += 1
                    if (cpt_1 + cpt_0) == len(colors_from_prompt.values()):
                        # image with all the cls associated with a color are detected
                        count_order[key][0] += 1
                        for i in range(1, len(count_order[key])):
                            count_order[key][i] += count_order_temp[key][i - 1]
                else:
                    results_for_prompt[key] = None

        results_2_save.append(results)
        if path_to_save_json is not None:
            for key in RESULT_KEYS:
                if key in results:
                    if results[key][0] == 1:
                        sorted_images[key]["correct"].append(
                            f"{prompt.replace(' ', '_')}_{seed}"
                        )
                    else:
                        sorted_images[key]["failure"].append(
                            f"{prompt.replace(' ', '_')}_{seed}"
                        )
            if binding:
                for key in KEYS_BINDING:
                    if key in results:
                        if results[key] == 1:
                            sorted_images[key]["correct"].append(
                                f"{prompt.replace(' ', '_')}_{seed}"
                            )
                        else:
                            sorted_images[key]["failure"].append(
                                f"{prompt.replace(' ', '_')}_{seed}"
                            )
    if path_to_save_json is not None:
        if lock_json is not None:
            lock_json.acquire()
        keys = RESULT_KEYS.copy()
        if binding:
            keys += KEYS_BINDING
        save2json(sorted_images, path_to_save_json, keys)
        if lock_json is not None:
            lock_json.release()
    for key in RESULT_KEYS:
        if key in results:
            results_for_prompt[key] /= nb_items
        else:
            results_for_prompt.pop(key)

    if binding:
        for key in KEYS_BINDING:
            if key in results:
                results_for_prompt[key] /= nb_items
            else:
                results_for_prompt.pop(key)

    for key in COUNT_KEYS:
        if key in count_order_temp:
            for i in range(len(count_order[key])):
                count_order[key][i] /= nb_items
        else:
            count_order.pop(key)

    params2save = {
        "conf_min": conf_min,
        "remove_superposed": remove_superposed,
        "remove_supervised_type": remove_supervised_type,
        "remove_superposed_iou_min": remove_superposed_iou_min,
        "keep_superposed_with_best_conf": keep_superposed_with_best_conf,
    }

    return_dict = {
        f"params_results_{name_new_column}": params2save,
        f"results_{name_new_column}": results_2_save,
        f"results_for_prompt_{name_new_column}": results_for_prompt,
        f"count_order_{name_new_column}": count_order,
    }
    if binding:
        return_dict[
            f"correct_binding_per_pos_of_class_associated_with_color_{name_new_column}"
        ] = correct_binding_per_pos_of_class_associated_with_color
    return return_dict


def filter_dataset(batch, filter_group, params2group):
    # cannot be batched
    if not all(p in params2group.keys() for p in batch.values()):
        return None
    classes = tuple(params2group[p] for p in batch.values())
    if classes == filter_group:
        return batch
    return None


def get_available_masks(masks):
    available_masks = set(("_".join(x.split("_")[:-1]) + "_") for x in masks.files)
    prompt2masks = {
        p: [x for x in masks.files if x.startswith(p)] for p in available_masks
    }
    # sort by seed
    sorted_prompt2masks = {
        key: sorted(value, key=lambda x: int(x.split("_")[-1]))
        for key, value in prompt2masks.items()
    }

    return available_masks, sorted_prompt2masks


def get_masks_name(available_masks, prompt2masks, prompt, seed):
    if f"{prompt.replace(' ', '_')}_mask_{seed}_" in available_masks:
        return prompt2masks[f"{prompt.replace(' ', '_')}_mask_{seed}_"]
    else:
        return None


def map_attribute_binding(
    batch,
    path_to_masks,
    images_per_prompt,
    tar_images,
):
    prompt = batch
    labels_distances = []
    percentages_distances = []

    images = []
    for seed in range(images_per_prompt):
        f = tar_images.extractfile(prompt.replace(" ", "_") + f"_{seed}.png")
        image = Image.open(io.BytesIO(f.read()))
        images.append(np.array(image))

    masks = np.load(f"{path_to_masks}/{prompt.replace(' ', '_')}.npz")
    available_masks, prompt2masks = get_available_masks(masks)

    for seed, img in zip(range(images_per_prompt), images):
        masks_name = get_masks_name(available_masks, prompt2masks, prompt, seed)
        if masks_name is not None:
            labels_temp = []
            percentages_temp = []
            for m in masks_name:
                # load mask
                l, p = eval_distance_to_colors(img, masks[m])
                labels_temp.append(l.tolist())
                percentages_temp.append(p.tolist())

            labels_distances.append(labels_temp)
            percentages_distances.append(percentages_temp)
        else:
            labels_distances.append([])
            percentages_distances.append([])

    results = {
        "labels_distances": labels_distances,
        "percentages_distances": percentages_distances,
    }
    return results


def open_tarball(path):
    if path.endswith(".tar"):
        return tarfile.open(path, "r")
    elif path.endswith(".tar.gz"):
        return tarfile.open(path, "r:gz")
    elif path.endswith(".tar.bz2"):
        return tarfile.open(path, "r:bz2")
    elif path.endswith(".tar.xz"):
        return tarfile.open(path, "r:xz")
    else:
        raise ValueError(f"Unknown extension for {path} or not a tarball")
