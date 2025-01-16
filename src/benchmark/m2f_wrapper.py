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
        hf_token=None,
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

        self.backbone, self.transform = self.create_backbone(self.model_name, backbone_size=self.backbone_size, hf_token=hf_token)
        self.backbone.eval()
        self.backbone.cuda()

        self.model = self.create_segmenter(cfg, backbone_model=self.backbone)

        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        state_dict = torch.hub.load_state_dict_from_url(head_checkpoint_url, map_location="cpu")
        print(head_checkpoint_url)
        #runner.load_checkpoint(self.model, head_checkpoint_url, map_location="CPU")
        # This doesn't work because of map_location, but okay bc uses torch.load in backend
        self.model.load_state_dict(state_dict['state_dict'], strict=True)
        self.model.cuda()
        self.model.eval()

    def create_backbone(self, model_name, backbone_size="small", hf_token=None):
        model_name = model_name.lower()
        if model_name == "dinov2":
            # backbone_size in ("small", "base", "large" or "giant")
            BACKBONE_ARCHS = {
                "small": "vits14",
                "base": "vitb14",
                "large": "vitl14",
                "giant": "vitg14",
            }
            backbone_arch = BACKBONE_ARCHS[backbone_size]
            backbone_name = f"dinov2_{backbone_arch}"

            backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
            transform = None

        elif model_name == "uni":
            login(hf_token)
            backbone_model = timm.create_model(
                "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True
            )
            transform = create_transform(**resolve_data_config(model=backbone_model))
        elif model_name == "schuerch":
            raise NotImplementedError(f"Model {model_name} is not implemented.")
        elif model_name == "virchow":
            raise NotImplementedError(f"Model {model_name} is not implemented.")
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented.")
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
        #b, c, h, w = x.shape
        x = self.transform(x)
        #in_array = x[:, :, ::-1]  # BGR
        #in_array = in_array.permute(0, 3, 1, 2)
        h, w, _ = x.shape
        x = x.unsqueeze(0)
        print('model fwd input shape', x.shape)
        #segmentation_logits = mmseg.apis.inference_model(self.model, in_array)[0]
        img_meta = [{"img_shape": (3, h, w), "ori_shape": (3, h, w), 'scale_factor': 1.0, 'flip': False}]
        segmentation_logits = self.model.inference(x, img_meta,rescale=True)
        return segmentation_logits

