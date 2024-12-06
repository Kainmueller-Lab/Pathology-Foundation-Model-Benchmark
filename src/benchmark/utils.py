from typing import Union, Tuple
import pandas as pd
import numpy as np
from benchmark.lmdb_dataset import LMDBDataset
import json


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


def prep_datasets(cfg):
    """Prepare the training and validation datasets.

    Args:
        cfg (omegaconf): The configuration object.

    Returns:
        tuple: A tuple containing the train, validation, test datasets, and the label dictionary.
    """
    # load split .csv
    split_df = pd.read_csv(cfg.dataset.split) 
    # enforce column names
    assert set(['sample_name', 'train_test_val_split']) == set(split_df.columns)
    # enfore split names
    assert set(['train', 'test', 'valid']) == set(split_df['train_test_val_split'].unique())
    label_dict = json.load(open(cfg.dataset.label_dict))
    # load dataset
    datasets = []
    for split in ['train', 'valid', 'test']:
        include_fovs = split_df[split_df['train_test_val_split'] == split]['sample_name'].tolist()
        dataset = LMDBDataset(path=cfg.dataset.path, include_sample_names=include_fovs)
        datasets.append(dataset)
    datasets.append(label_dict)
    return tuple(datasets)
