from typing import Union, Tuple
import numpy as np
from PIL import Image


def get_height_width(array):
    """Get the height and width of the input array.

    Args:
        array (np.ndarray): A NumPy array with 2 or 3 dimensions.

    Returns:
        int, int: The height and width of the array.

    Raises:
        ValueError: If the input array has an invalid number of dimensions.
    """
    # Get the dimensions of the image
    if np.ndim(array) == 2:
        height, width = array.shape
    elif np.ndim(array) == 3:
        _, height, width = array.shape
    else:
        raise ValueError("Input array must have 2 or 3 dimensions.")
    return height, width


def to_tuple(dim: Union[int, tuple[int, ...]]) -> tuple[int, ...]:
    """Convert a dimension to a tuple.

    Args:
        dim (int or tuple of int): The input dimension.

    Returns:
        tuple of int: The dimension as a tuple of two integers, or
        the tuple itself.
    """
    if isinstance(dim, int):
        return (dim, dim)
    elif isinstance(dim, tuple):
        return dim
    else:
        raise ValueError("Dimension must be an integer or a tuple of two integers.")


def render_segmentation(segmentation_logits, dataset):
    raise NotImplementedError
    # TODO: Define colormaps

    DATASET_COLORMAPS = None

    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)
