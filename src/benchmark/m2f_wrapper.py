import os
import torch
import math
import itertools
import mmcv
import torch
import timm
import mmseg


from functools import partial
import torch.nn.functional as F
from mmengine import runner, config
from huggingface_hub import login
import numpy as np

from torchmetrics.classification import MulticlassJaccardIndex
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config

from backbones import encoder_decoder_m2f
from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.pannuke import PanNukeDataset
from bio_image_datasets.consep_dataset import ConSePDataset
from bio_image_datasets.schuerch_dataset import SchuerchDataset
from bio_image_datasets.segpath_dataset import SegPath

from mmengine.runner import Runner

from benchmark.simple_segmentation_model import clean_str, load_model_and_transform

BACKBONE_ARCHS = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class M2FSegmentationModel(torch.nn.Module):
    def __init__(
        self,
        backbone_size,
        model_name,
        num_classes,
        head_checkpoint_url,
        cfg_str,
        head_type="ms",
        head_scale_count=3,
    ):
        super().__init__()
        self.head_type = head_type
        self.head_scale_count = head_scale_count
        self.num_classes = num_classes
        self.model_name = model_name
        self.backbone_size = backbone_size

        # Ensure that cfg_str is a valid path or content for a configuration file
        cfg = config.Config.fromfile(cfg_str)
        if self.head_type == "ms":
            cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][: self.head_scale_count]
            print("scales:", cfg.data.test.pipeline[1]["img_ratios"])
        
        h, w = cfg.crop_size
        c = cfg.img_channels
        self.img_meta = [{"img_shape": (h, w, c), "ori_shape": (h, w, c),
                          'scale_factor': cfg.data.test.pipeline[1]["img_ratios"],
                          'flip': cfg.data.test.pipeline[1]["flip"],
                          'flip_direction': cfg.data.test.pipeline[1]["flip_direction"],}]
        self.backbone, self.transform = self.create_backbone(self.model_name, backbone_size=self.backbone_size)
        self.backbone.eval()
        self.backbone.cuda()

        self.model = self.create_segmenter(cfg, backbone_model=self.backbone)

        state_dict = torch.hub.load_state_dict_from_url(head_checkpoint_url, map_location="cpu")
        print(f"Using head_checkpoint_url: {head_checkpoint_url}")

        #runner.load_checkpoint(self.model, head_checkpoint_url, map_location="CPU")
        # This doesn't work because of map_location, but okay bc uses torch.load in backend
        # decode_head cannot be loaded because the num_classes might change
        state_dict = {key: val for key, val in state_dict['state_dict'].items() if 'decode_head.conv_seg.' not in key}

        self.model.load_state_dict(state_dict, strict=False)
        self.model.cuda()
        self.model.eval()

    def create_backbone(self, model_name):
        model_name = clean_str(model_name)
        if 'dinov2' in model_name:
            # model_name in ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"):
            backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name)
            transform = None
        else:
            backbone_model, transform, model_dim = load_model_and_transform(model_name)
        print(f'Backbone model {model_name} loaded with dims {model_dim}')
        return backbone_model, transform

    def create_segmenter(self, cfg, backbone_model):
        rel_output_path = os.path.join('..','..','outputs')
        os.makedirs(rel_output_path, exist_ok=True)
        runner = Runner.from_cfg(cfg)

        runner.model.backbone.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
        )

        if hasattr(backbone_model, "patch_size"):
            print('backbone_model.patch_size', backbone_model.patch_size)
            runner.model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
        runner.model.init_weights()

        return runner.model

    def forward(self, x):
        x = self.transform(x)
        x = x.unsqueeze(0)
        #print('model fwd input shape', x.shape, self.model.inference, self.model)

        segmentation_logits = self.model.inference(x, self.img_meta, rescale=True)
        return segmentation_logits

