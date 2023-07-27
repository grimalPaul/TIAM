import cv2
import numpy as np

from .colors import comparator_Berlin_Kay_lab, sRGB_to_lab


def cluster_with_distance(pixels, comparator):
    matrix = np.zeros(shape=(pixels.shape[0], len(comparator)))
    for i, values in enumerate(comparator.values()):
        matrix[:, i] = np.linalg.norm(pixels - values, ord=2, axis=1)
    labels, counts = np.unique(np.argmin(matrix, axis=1), return_counts=True)
    indices = np.argsort(counts)[::-1]
    labels = np.take_along_axis(labels, indices, axis=0)
    counts = np.take_along_axis(counts, indices, axis=0)
    labels = np.array(list(comparator.keys()))[labels]
    percentages = np.array(counts) / pixels.shape[0]
    return labels, percentages


def eval_distance_to_colors(image, mask, n=5):
    """
    return the n closest colors to the image_xyz

    image: np.array of shape (h, w, 3)
    mask: np.array
    """
    mask_size = image.shape[:2]
    img = sRGB_to_lab(image / 255)
    img = img.reshape(-1, 3)

    # select only pixels that are in the mask
    binary_mask = segment2mask(
        mask,
        mask_shape=mask_size,
    ).reshape(-1)
    to_keep = np.where(binary_mask == 1)[0]
    img = img[to_keep]
    # compute closest colors
    labels, percentages = cluster_with_distance(img, comparator_Berlin_Kay_lab)
    return labels[:n], percentages[:n]


def segment2mask(segment, mask_shape, normalized=False):
    """
    Convert a segment (list of (x,y) points) to a binary mask of the specified shape.
    Args:
      segment (np.ndarray): the segment mask as a numpy array of shape (n,2)
      mask_shape (tuple): the desired shape of the output binary mask as a tuple of (height, width)
      normalized (bool): whether the segment is normalized to [0, 1] or not
    Returns:
      mask (np.ndarray): the binary mask as a numpy array of shape (height, width)
    """
    h, w = mask_shape
    if normalized:
        coords = np.round(segment * np.array([w, h])).astype(int)
    else:
        coords = segment.astype(int)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    if len(coords) > 0:
        # Fill the mask
        cv2.fillPoly(mask, [coords], 255)
    mask[mask == 255] = 1
    return mask
