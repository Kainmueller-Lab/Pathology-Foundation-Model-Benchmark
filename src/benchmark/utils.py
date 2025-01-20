from typing import Union, Tuple
import pandas as pd
import numpy as np
from benchmark.lmdb_dataset import LMDBDataset
import json
import torch
import torch.nn as nn
from typing import Union
import matplotlib.pyplot as plt

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


def exclude_classes(
        exclude_classes: Union[int, list[int]], loss: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exclude specified classes from the loss calculation.

    Args:
        exclude_classes (int or list of ints): Classes to exclude from the loss calculation.
        loss (torch.Tensor): The calculated loss.
        target (torch.Tensor): The target tensor.
    
    Returns:
        tuple: A tuple containing the modified loss and mask tensors.
    """
    mask = torch.ones_like(loss)
    for c in exclude_classes:
        loss[target == c] = 0
        mask[target == c] = 0
    return loss, mask


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
        loss, mask = exclude_classes(self.exclude_class, loss, target)
        return loss.sum() / mask.sum()


class EMAInverseClassFrequencyLoss(nn.Module):
    """A loss wrapper that uses exponential moving average (EMA) of the inverse class frequency
    for loss weighting and excludes specified classes.

    Args:
        loss_fn (nn.Module): The base loss function.
        exclude_class (int or list of ints): Classes to exclude from the loss calculation.
        num_classes (int): Total number of classes in the dataset.
        alpha (float): Smoothing factor for EMA, default is 0.99.
    """
    def __init__(
        self, loss_fn: nn.Module, exclude_class: Union[int, list[int]], num_classes: int,
        alpha: float = 0.99, class_weighting=False
        ):
        super().__init__()
        self.loss_fn = loss_fn
        self.exclude_class = exclude_class
        self.num_classes = num_classes
        self.alpha = alpha
        # Initialize EMA frequencies with small non-zero values to avoid division by zero
        self.ema_frequencies = torch.ones(num_classes) * 1e-6
        self.class_weighting = class_weighting

    def update_frequencies(self, target: torch.Tensor):
        """Update EMA frequencies based on the target tensor.

        Args:
            target (torch.Tensor): The target tensor containing class labels.
        """
        with torch.no_grad():
            class_counts = torch.bincount(target.flatten(), minlength=self.num_classes).float().cpu()
            self.ema_frequencies = self.alpha * self.ema_frequencies + (1 - self.alpha) * class_counts

    def calculate_weights(self):
        """Calculate inverse class frequency weights from EMA frequencies.

        Returns:
            torch.Tensor: The weights for each class.
        """
        inverse_frequencies = 1.0 / (self.ema_frequencies + 1e-6)
        return inverse_frequencies / inverse_frequencies.sum()  # Normalize weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the weighted loss, excluding specified classes.

        Args:
            pred (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The calculated loss.
        """
        if self.class_weighting:
            # Update EMA frequencies with the current target
            self.update_frequencies(target)

            # Calculate class weights
            weights = self.calculate_weights()
            for c in self.exclude_class:
                weights[c] = 0
            
            self.loss_fn.weight = weights.to(pred.device).softmax(dim=0)

        # Apply weights to loss
        loss = self.loss_fn(pred, target)
        loss, mask = exclude_classes(self.exclude_class, loss, target)
        return loss.sum() / mask.sum()


def render_segmentation(segmentation_classes, num_classes=None):
    seg_img = np.zeros((segmentation_classes.shape[0], segmentation_classes.shape[1], 4))
    cmap = plt.get_cmap('hot')
    for i in range(num_classes):
        idxs = segmentation_classes == i
        seg_img[idxs] = cmap(i / num_classes)

    return seg_img
