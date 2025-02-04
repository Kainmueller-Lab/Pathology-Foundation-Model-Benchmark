from logging import config
import os
from omegaconf import OmegaConf
import torch

import torch.nn.functional as F
import numpy as np


from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.pannuke import PanNukeDataset
from bio_image_datasets.consep_dataset import ConSePDataset
from bio_image_datasets.schuerch_dataset import SchuerchDataset
from bio_image_datasets.segpath_dataset import SegPath

from benchmark.simple_segmentation_model import clean_str, load_model_and_transform


from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


BACKBONE_ARCHS = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}


class M2F_HF_SegmentationModel(torch.nn.Module):
    def __init__(
        self,
        backbone_size,
        model_name,
        num_classes,
        cfg_str,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.backbone_size = backbone_size

        # Ensure that cfg_str is a valid path or content for a configuration file
        cfg = OmegaConf.load(cfg_str)

        self.height = self.width = cfg.crop_size
        self.num_channels = cfg.img_channels

        self.backbone, self.transform = self.create_backbone(self.model_name)
        self.backbone.cuda()
        self.backbone.eval()

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        self.model.cuda()
        self.model.eval()

    def create_backbone(self, model_name):
        model_name = clean_str(model_name)
        if "dinov2" in model_name:
            # model_name in ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"):
            backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name)
            transform = None
        else:
            backbone_model, transform, model_dim = load_model_and_transform(model_name)
        print(f"Backbone model {model_name} loaded with dims {model_dim}")
        return backbone_model, transform

    def forward(self, x):
        print(x.keys(), x["pixel_values"].shape, x["pixel_values"].device)
        # b, c, h, w = x[0].shape
        # inputs = self.image_processor(x, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**x)

        pred_semantic_map = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(self.height, self.width)]
        )[0]
        return pred_semantic_map
