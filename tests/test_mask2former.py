import os
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from benchmark.eval import Eval, extract_numbers_from_string
from benchmark.m2f_wrapper import M2FSegmentationModel

BACKBONE_ARCHS = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}


def test_instantiate_model():
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    backbone_size = "small"
    backbone_arch = BACKBONE_ARCHS[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"
    head_scale_count = 4  # more scales: slower but better results, in (1,2,3,4,5)
    head_dataset = "voc2012"  # in ("ade20k", "voc2012")
    head_type = "ms"  # in ("ms, "linear")

    num_classes = 8

    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_head.pth"
    head_config_file = f"./config/{backbone_name}_{head_dataset}_{head_type}_config.py"

    model_name = 'provgigapath'

    model = M2FSegmentationModel(
        model_name=model_name,
        num_classes=num_classes,
        head_checkpoint_url=head_checkpoint_url,
        cfg_str=head_config_file,
        head_type=head_type,
        head_scale_count=head_scale_count,
    )

    assert isinstance(model, M2FSegmentationModel)

