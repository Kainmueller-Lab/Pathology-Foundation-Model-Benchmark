from typing import Union, Tuple
import pandas as pd
import numpy as np
from benchmark.lmdb_dataset import LMDBDataset
import json
import torch
import torch.nn as nn
from typing import Union


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


class ExcludeClassLossWrapper(nn.Module):
    """A loss wrapper that excludes specified classes from the loss calculation.

    Args:
        loss_fn (nn.Module): The loss function to wrap.
        exclude_class (int or list of ints): Classes to exclude from the loss calculation.
    """
    def __init__(self, loss_fn: nn.Module, exclude_class: Union[int, list[int]]):
        super().__init__()
        self.loss_fn = loss_fn
        if isinstance(exclude_class, int):
            self.exclude_class = [exclude_class]
        else:
            self.exclude_class = exclude_class

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss, excluding specified classes.
        
        Args:
            pred (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The target tensor.
        
        Returns:
            torch.Tensor: The calculated loss.
        """
        # Calculate the loss only for included classes
        loss = self.loss_fn(pred, target)
        mask = torch.ones_like(loss)
        for c in self.exclude_class:
            loss[target == c] = 0
            mask[target == c] = 0
        return loss.sum() / mask.sum()


def render_segmentation(segmentation_logits, dataset):
    raise NotImplementedError
    # TODO: Define colormaps

    DATASET_COLORMAPS = None

    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)
