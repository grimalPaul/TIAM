import itertools
from typing import List

import numpy as np

from .attribute_binding import segment2mask
from .coco_data import getlabels2numbers


def all_classes_detected(classes_from_prompt: List[str], detected_classes: List[int]):
    """
    Detect if at least each class from the prompt is detected one time in the image
    If in the prompt there is "a and a", it will be considered detected even if there is only one "a" detected
    Args:
        classes_from_prompt: list of classes from the prompt
        detected_classes: list of detected classes
    Returns:
        all_classes_detected: 1 if all classes are detected, 0 otherwise
        number of classes detected
    """
    labels2number = getlabels2numbers()
    classes_prompt = [labels2number[p] for p in classes_from_prompt]
    # remove duplicates
    detected_classes = list(set(detected_classes))
    # In case of sentence like "a and a"
    classes_prompt = list(set(classes_prompt))
    # compare the two lists and give a score of how many classes are detected / how many classes must be in the image
    len_classes_prompt = len(classes_prompt)
    score = 0
    for c in detected_classes:
        if c in classes_prompt:
            score += 1
    return 1 if score == len_classes_prompt else 0, score


def compute_iou_bbox(bbox1, bbox2):
    """Compute the IoU between two bboxs"""
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def compute_iou_seg(mask1: np.array, mask2: np.array) -> float:
    """Compute the IoU between two segmentation masks

    Args:
        mask1 (np.array): segmentation mask of shape (h, w)
        mask2 (np.array): segmentation mask of shape (h, w)

    Returns:
        float: IoU
    """
    intersection = (mask1 * mask2).sum()
    union = mask1.sum() + mask2.sum() - intersection
    return (intersection / (union)).item()


def multi_combinations(objects):
    # [['a','b'],['c','d']]
    # ('a', 'c'), ('a', 'd')
    # ('b', 'c'),('b', 'd')
    if len(objects) > 2:
        raise ValueError("We can only compare two objects at a time")
    return itertools.product(*objects)


def compare_order_in_prompt_with_detected_classes(
    classes_from_prompt: List[str],
    detected_classes,
):
    """
    Compare the order of the classes in the prompt with the classes that are detected
    i.e. if the prompt is "a and b" and the detected classes are for 2 images are [a, b] and [a]
    The score will be [1, 0.5], we detect better element a than b
    """
    labels2number = getlabels2numbers()
    classes_prompt = [labels2number[p] for p in classes_from_prompt]
    count = [0] * len(classes_prompt)
    detected_classes = detected_classes.tolist()
    for i, c in enumerate(classes_prompt):
        if c in detected_classes:
            count[i] += 1
    return count


def compute_binding_score(
    detected_classes,
    detected_colors,
    percentage_colors,
    colors_to_detect,
    threshold_colors,
    classes_with_color,
):
    """Compute TIAM

    Returns:
        correct_bind: 1 if all classes with color are detected and the color is correct, 0 otherwise
        count: list of 0 and 1, 1 if the class with color is detected and the color is correct, 0 otherwise

    """
    correct = {}
    score_is_null = False
    for c in classes_with_color:
        if c in detected_classes:
            # check if it is the right color
            correct[c] = 0
            idx_c = np.where(detected_classes == c)[0]
            for i in idx_c:
                if (
                    colors_to_detect[c] in detected_colors[i]
                    and percentage_colors[i][
                        np.where(detected_colors[i] == colors_to_detect[c])[0]
                    ]
                    >= threshold_colors
                ):
                    correct[c] = 1
                    break
        else:
            score_is_null = True

    if len(correct) > 0 and not score_is_null:
        # all cls with color detected and correct bind
        if sum(correct.values()) == len(correct):
            correct_bind = 1
        else:
            correct_bind = 0
    else:
        correct_bind = 0
    count = [None] * len(classes_with_color)
    for i, c in enumerate(classes_with_color):
        if c in correct.keys():
            if correct[c]:
                count[i] = 1
            else:
                count[i] = 0
    return correct_bind, count


def to_homogeneous_array(array: List[List], add_value=np.nan):
    """Add value to the array to make it homogeneous

    Args:
        array (List[List]): list of list
        add_value (_type_, optional): Defaults to np.nan.

    Returns:
        homogeneous_array: list of list with the same length
    """
    max_len = 0
    for i in range(len(array)):
        if len(array[i]) > max_len:
            max_len = len(array[i])
    homogeneous_array = []
    for array_ in array:
        if len(array_) < max_len:
            homogeneous_array.append(array_ + [add_value] * (max_len - len(array_)))
        else:
            homogeneous_array.append(array_)
    return homogeneous_array


def eval(
    classes_from_prompt: List[str],
    detected_classes: List[int],
    bbox: List[List[float]],
    conf: List[float],
    masks=None,
    masks_name=None,
    conf_min=0.5,
    remove_superposed=True,
    remove_superposed_iou_min=0.95,
    remove_superposed_type="bbox",  # "seg"
    keep_superposed_with_best_conf=False,
    mask_size=(512, 512),
    binding=False,
    detected_colors=None,
    percentage_colors=None,
    colors_from_prompt=None,
    threshold_colors=0.4,
):
    """
    Args:
        classes_from_prompt: list of classes from the prompt
        detected_classes: list of detected classes
        bbox: list of bbox
        conf: list of confidence
        masks: numpy array of masks
        conf_min: minimum confidence to keep a detection
        remove_superposed: if True, remove superposed bboxs/segs
        remove_superposed_iou_min: minimum IoU to consider two bboxs/seg superposed
        remove_superposed_type: "bbox" or "seg" or ["bbox", "seg"]
        keep_superposed_with_best_conf: if True, keep only the bbox with the best confidence
        mask_size: size of the mask
        binding: if True, we measure the attribute binding
        detected_colors: list of colors detected in the masks
        percentage_colors: percentage of each color detected in the masks
        colors_from_prompt: the labels in the prompt with associated color
        threshold_colors: threshold to consider a color as detected
    Returns:
        results: dictionary with the results
            results["without_filter"]
            results["with_conf_min"] apply a confidence threshold
            results["conf_min_bbox"] "with_conf_min" and remove superposed bboxs
            results["conf_min_seg"] "with_conf_min" and remove superposed segs

        count_order: dictionary with the represenation of each class on the image according to the position in the prompt
            see compare_order_in_prompt_with_detected_classes
            count_order["without_filter"]
            count_order["with_conf_min"] apply a confidence threshold
            count_order["conf_min_bbox"] "with_conf_min" and remove superposed bboxs
            count_order["conf_min_seg"] "with_conf_min" and remove superposed segs
    """

    results = {}
    count_order = {}
    bbox = np.array(bbox)
    detected_classes = np.array(detected_classes)
    conf = np.array(conf)
    if masks is not None and masks_name is not None:
        mask_from_segment = []
        for mask in masks_name:
            m = masks[mask]
            mask_from_segment.append(
                segment2mask(
                    np.array(m),
                    mask_shape=mask_size,
                )
            )
        masks = np.stack(mask_from_segment)
    else:
        masks = None
    if binding:
        labels2number = getlabels2numbers()
        colors_to_detect = {
            labels2number[label]: c for label, c in colors_from_prompt.items()
        }
        classes_with_color = list(colors_to_detect.keys())

        detected_colors = np.array(to_homogeneous_array(detected_colors))
        percentage_colors = np.array(to_homogeneous_array(percentage_colors))

    classes_from_prompt = classes_from_prompt.values()
    results["without_filter"] = all_classes_detected(
        classes_from_prompt, detected_classes
    )

    count_order["without_filter"] = compare_order_in_prompt_with_detected_classes(
        classes_from_prompt=classes_from_prompt, detected_classes=detected_classes
    )

    if binding:
        (
            results["without_filter_binding"],
            count_order["without_filter_binding"],
        ) = compute_binding_score(
            classes_with_color=classes_with_color,
            colors_to_detect=colors_to_detect,
            detected_classes=detected_classes,
            detected_colors=detected_colors,
            percentage_colors=percentage_colors,
            threshold_colors=threshold_colors,
        )

    # remove if confidence is too low
    if conf_min is not None:
        indices_to_keep = np.where(conf >= conf_min)[0]
        bbox = bbox[indices_to_keep]
        detected_classes = detected_classes[indices_to_keep]
        if masks is not None:
            masks = masks[indices_to_keep]
        conf = conf[indices_to_keep]
        results["with_conf_min"] = all_classes_detected(
            classes_from_prompt, detected_classes
        )
        count_order["with_conf_min"] = compare_order_in_prompt_with_detected_classes(
            classes_from_prompt=classes_from_prompt, detected_classes=detected_classes
        )
        if binding:
            detected_colors = detected_colors[indices_to_keep]
            percentage_colors = percentage_colors[indices_to_keep]
            (
                results["with_conf_min_binding"],
                count_order["with_conf_min_binding"],
            ) = compute_binding_score(
                classes_with_color=classes_with_color,
                colors_to_detect=colors_to_detect,
                detected_classes=detected_classes,
                detected_colors=detected_colors,
                percentage_colors=percentage_colors,
                threshold_colors=threshold_colors,
            )
    # remove superposed
    if remove_superposed:
        if isinstance(remove_superposed_type, str):
            remove_superposed_type = [remove_superposed_type]

        classes2index = {
            val: [i for i, x in enumerate(detected_classes) if x == val]
            for val in detected_classes
        }
        classes = itertools.combinations(classes2index.keys(), 2)
        indices_to_check_bbox = []
        indices_to_check_seg = []
        for c1, c2 in classes:
            iterable = multi_combinations([classes2index[c1], classes2index[c2]])
            for i, j in iterable:
                for t in remove_superposed_type:
                    if t == "bbox":
                        iou_bbox = compute_iou_bbox(bbox[i], bbox[j])
                        if iou_bbox >= remove_superposed_iou_min:
                            indices_to_check_bbox.append((i, j))
                    if t == "seg":
                        iou_seg = compute_iou_seg(masks[i], masks[j])
                        if iou_seg >= remove_superposed_iou_min:
                            indices_to_check_seg.append((i, j))

                    if t not in ["bbox", "seg"]:
                        raise ValueError(
                            f"remove_superposed_type must be 'bbox' or 'seg', not {remove_superposed_type}"
                        )
        # process indices
        indices_to_check_bbox = list(set(indices_to_check_bbox))
        indices_to_check_seg = list(set(indices_to_check_seg))

        for t in remove_superposed_type:
            if t == "bbox":
                indices_to_rm_bbox = list(
                    set(
                        indices_to_remove(
                            indices_to_check_bbox,
                            conf,
                            keep_superposed_with_best_conf,
                            classes_from_prompt,
                            t,
                        )
                    )
                )
                detected_classes_bbox = np.delete(detected_classes, indices_to_rm_bbox)
                results[f"conf_min_{t}"] = all_classes_detected(
                    classes_from_prompt, detected_classes_bbox
                )
                count_order[
                    f"conf_min_{t}"
                ] = compare_order_in_prompt_with_detected_classes(
                    classes_from_prompt=classes_from_prompt,
                    detected_classes=detected_classes_bbox,
                )

            else:
                indices_to_rm_seg = list(
                    set(
                        indices_to_remove(
                            indices_to_check_seg,
                            conf,
                            keep_superposed_with_best_conf,
                            classes_from_prompt,
                            t,
                        )
                    )
                )
                detected_classes_seg = np.delete(detected_classes, indices_to_rm_seg)
                results[f"conf_min_{t}"] = all_classes_detected(
                    classes_from_prompt, detected_classes_seg
                )
                count_order[
                    f"conf_min_{t}"
                ] = compare_order_in_prompt_with_detected_classes(
                    classes_from_prompt=classes_from_prompt,
                    detected_classes=detected_classes_seg,
                )

                if binding:
                    detected_colors_seg = np.delete(
                        detected_colors, indices_to_rm_seg, axis=0
                    )
                    percentage_colors_seg = np.delete(
                        percentage_colors, indices_to_rm_seg, axis=0
                    )
                    (
                        results[f"conf_min_{t}_binding"],
                        count_order[f"conf_min_{t}_binding"],
                    ) = compute_binding_score(
                        classes_with_color=classes_with_color,
                        colors_to_detect=colors_to_detect,
                        detected_classes=detected_classes_seg,
                        detected_colors=detected_colors_seg,
                        percentage_colors=percentage_colors_seg,
                        threshold_colors=threshold_colors,
                    )

    return results, count_order


def indices_to_remove(
    indices_to_check,
    conf,
    keep_superposed_with_best_conf,
    classes_from_prompt,
    remove_superposed_type,
):
    """Compute the indices to remove"""
    indices_to_rm = []
    for i, j in indices_to_check:
        if keep_superposed_with_best_conf:
            if conf[i] > conf[j]:
                indices_to_rm.append(j)
            elif conf[i] < conf[j]:
                indices_to_rm.append(i)
            else:
                print(
                    f"Two {remove_superposed_type} of 2 different classes \
                    have the same shape & same confident.\n \
                    We keep both of them. \n \
                    boxes {i} and {j}, params {classes_from_prompt} "
                )
        else:
            indices_to_rm.append(j)
            indices_to_rm.append(i)
    return indices_to_rm
