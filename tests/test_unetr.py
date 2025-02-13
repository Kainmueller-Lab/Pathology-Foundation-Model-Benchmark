from benchmark.unetr import UnetR
import numpy as np
from omegaconf import OmegaConf
import torch


def test_unetr():
    image = torch.rand(2, 3, 224, 224)
    model_name = "mock"
    num_classes = 8
    model = UnetR(
        model_name=model_name,
        num_classes=num_classes,
    )
    output = model(image)
    assert output.shape == (2, num_classes, 224, 224)
