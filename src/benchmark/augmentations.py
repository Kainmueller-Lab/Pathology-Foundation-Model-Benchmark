import kornia.augmentation as Kaug
from benchmark import custom_augmentations as Caug
import warnings
warnings.simplefilter('once', UserWarning)  # Set this at the script start


def get_augmentation(name, **kwargs):
    """Get the augmentation class from the Kornia or custom augmentations.
    Args:
        name (str): Name of the augmentation class.
        kwargs (dict): Keyword arguments for the augmentation class.
    Returns:
        Kornia or custom augmentation class.
    """
    if hasattr(Caug, name):
        return getattr(Caug, name)(**kwargs)
    else:
        return getattr(Kaug, name)(**kwargs)


class Augmenter(Kaug.AugmentationSequential):
    """Augmenter class to apply augmentations to a batch of samples.
    """
    def __init__(self, params, **kwargs):
        """Initializes the Augmenter class.
        Args:
            params (dict): Dictionary of augmentation names and their parameters, e.g.
                params = {
                    "RandomHorizontalFlip": {"p": 0.5},
                    "RandomVerticalFlip": {"p": 1.0},
                }
            kwargs (dict): Keyword arguments for the transformations base class, e.g.
                data_keys=["image", "mask"], same_on_batch=False, keepdim=True, etc.
        """
        self.params = params
        self.transforms = self.define_augmentations()
        super(Augmenter, self).__init__(*self.transforms, **kwargs)

    def define_augmentations(self):
        """Creates the transformations based on names and values in params.
        """
        # creates the transformations based on names and values in params
        transforms = []
        for name, kwargs in self.params.items():
            transforms.append(get_augmentation(name, **kwargs))
        return transforms

    def repeat_last_transform(self, *args):
        """Repeats transformations with the same random parameters that were used in the last
        transform. Only works for certain transformations, e.g. noise distributions will not be
        identical.
        Args:
            args (dict): Keyword arguments for the last transform.
        """
        return self.__call__(*args, params=self._params)
