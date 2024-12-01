from abc import ABC, abstractmethod
from typing import Callable, Optional


class Dataset(ABC):
    """Abstract class to define the structure of a dataset.

    Args:
        local_path (str): Path to the dataset.
        transform (Optional[Callable]): Transform to apply to the data.
    """
    def __init__(self, local_path: str, transform: Optional[Callable] = None):
        self.local_path = local_path
        self.transform = transform

    @abstractmethod
    def download(self):
        """Download the dataset if it's not available in the local path."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Return the length / number of samples of the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """Return the item at the given index."""
        raise NotImplementedError

    @abstractmethod
    def load_sample(self, idx):
        """Load an individual sample by index."""
        raise NotImplementedError

    @abstractmethod
    def load_mask(self, idx):
        """Load the associated mask for a given sample by index."""
        raise NotImplementedError

    def __repr__(self):
        """Return the string representation of the dataset."""
        return self.__class__.__name__ + ' (' + self.local_path + ')'