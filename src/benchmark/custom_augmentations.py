import numpy as np
import torch
import kornia.augmentation as K
from typing import Optional


RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32
)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def torch_rgb2hed(img, hed_t, e):
    """Converts an RGB image to HED color space.
    
    Args:
        img (torch.Tensor): Image in RGB color space.
        hed_t (torch.Tensor): Transformation tensor.
        e (torch.Tensor): Small value to avoid log(0).

    Returns:
        torch.Tensor: Image in HED color space.
    """
    img = img.movedim(-3, -1)
    img = torch.clamp(img, min=e)
    img = torch.log(img) / torch.log(e)
    img = torch.matmul(img, hed_t)
    return img.movedim(-1, -3)


def torch_hed2rgb(img, rgb_t, e):
    """Converts an HED image to RGB color space.

    Args:
        img (torch.Tensor): Image in HED color space.
        rgb_t (torch.Tensor): Transformation tensor.
        e (torch.Tensor): Small value to avoid exp(-x).

    Returns:
        torch.Tensor: Image in RGB color space.
    """
    e = -torch.log(e)
    img = img.movedim(-3, -1)
    img = torch.matmul(-(img * e), rgb_t)
    img = torch.exp(img)
    img = torch.clamp(img, 0, 1)
    return img.movedim(-1, -3)


class HEDNormalize(K.IntensityAugmentationBase2D):
    def __init__(
        self, sigma: float = 0.05, bias: float = 0.1, same_on_batch: bool = False, 
        keepdim: bool = False,
    ):
        """Transforms RGB images to HED color space and adds augmentations in HED space before
            converting the images back to RGB space.

        Args:
            sigma (float): Maximum value for the random color shift.
            bias (float): Maximum value for the random color bias.
            same_on_batch (bool): Apply the same transformation across the batch.
            keepdim (bool): Keep the output shape the same as the input.
        
        Returns:
            torch.Tensor: Normalized image in RGB color space.
        """
        super().__init__(p=1.0, same_on_batch=same_on_batch, keepdim=keepdim)
        self.sigma = sigma
        self.bias = bias
        self.e = torch.tensor(1e-6)
        self.rgb_t = torch.from_numpy(RGB_FROM_HED)
        self.hed_t = torch.from_numpy(HED_FROM_RGB)

    def to_device(self, device):
        """Move the tensors to the specified device.

        Args:
            device (torch.device): Device to move the tensors to.
        """
        self.e = self.e.to(device)
        self.rgb_t = self.rgb_t.to(device)
        self.hed_t = self.hed_t.to(device)

    def rng(self, val, batch_size):
        """Generate random values for color shift and bias.

        Args:
            val (float): Maximum value for the random color shift and bias.
            batch_size (int): Number of random values to generate.

        Returns:
            torch.Tensor: Random values for color shift and bias.
        """
        return torch.empty(batch_size, 3).uniform_(-val, val)

    def color_norm_hed(self, img, sigmas, biases):
        """Apply color normalization in HED space.

        Args:
            img (torch.Tensor): Image in HED color space.
            sigmas (torch.Tensor): Random values for color shift.
            biases (torch.Tensor): Random values for color bias.

        Returns:
            torch.Tensor: Normalized image in HED color space.
        """
        return (img * (1 + sigmas.view(*sigmas.shape, 1, 1))) + biases.view(
            *biases.shape, 1, 1
        )

    def apply_transform(
            self, input: torch.Tensor, params: dict, flags, transform: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Apply the transformation to the input tensor.

        Args:
            input (torch.Tensor): Input tensor to transform.
            params (dict): Parameters for the transformation.
            flags: Flags for the transformation.
            transform (torch.Tensor): Transformation tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        batch_size = input.shape[0]
        device = input.device
        sigmas = self.rng(self.sigma, batch_size).to(device)
        biases = self.rng(self.bias, batch_size).to(device)
        if input.device != self.rgb_t.device:
            self.to_device(device)

        hed = torch_rgb2hed(input, self.hed_t, self.e)
        hed = self.color_norm_hed(hed, sigmas, biases)
        output = torch_hed2rgb(hed, self.rgb_t, self.e)
        return output
