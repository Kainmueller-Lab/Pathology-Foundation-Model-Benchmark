import os
import bio_image_datasets
from benchmark import utils
import torch
import math
import itertools
import mmcv
import torch
import timm
import mmseg

from functools import partial
import torch.nn.functional as F
from mmcv import runner
from torchmetrics.classification import MulticlassJaccardIndex
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config

from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.pannuke import PanNukeDataset
from bio_image_datasets.consep_dataset import ConSePDataset
from bio_image_datasets.schuerch_dataset import SchuerchDataset
from bio_image_datasets.segpath_dataset import SegPath

# import dinov2.eval.segmentation.models

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
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
        if self.head_type == "ms":
            cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][: self.head_scale_count]
            print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

        self.backbone, self.transform = self.lcreate_backbone(self.model_name, backbone_size=self.backbone_size)
        self.backbone.eval()
        self.backbone.cuda()

        self.model = self.create_segmenter(cfg, backbone_model=self.backbone)
        runner.load_checkpoint(self.model, head_checkpoint_url, map_location="CPU")
        self.model.cuda()
        self.model.eval()

    def create_backbone(self, model_name, backbone_size="small"):
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
            backbone_model = timm.create_model(
                "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True
            )
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        elif model_name == "schuerch":
            raise NotImplementedError(f"Model {model_name} is not implemented.")
        elif model_name == "virchow":
            raise NotImplementedError(f"Model {model_name} is not implemented.")
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented.")
        return backbone_model, transform

    def create_segmenter(self, cfg, backbone_model):
        model = mmseg.apis.init_segmentor(cfg)
        model.backbone.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
        )

        if hasattr(backbone_model, "patch_size"):
            model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
        model.init_weights()

        return model

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.transform(x)
        in_array = np.array(x)[:, :, ::-1]  # BGR
        segmentation_logits = mmseg.apis.inference_segmentor(self.model, in_array)[0]
        return segmentation_logits


if __name__ == "main":
    import numpy as np
    import matplotlib.pyplot as plt

    num_classes = 8

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    backbone_size = "small"
    backbone_arch = BACKBONE_ARCHS[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"
    head_scale_count = 3  # more scales: slower but better results, in (1,2,3,4,5)
    head_dataset = "voc2012"  # in ("ade20k", "voc2012")
    head_type = "ms"  # in ("ms, "linear")

    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_head.pth"

    model = M2FSegmentationModel(
        backbone_size=backbone_size,
        model_name="UNI",
        num_classes=num_classes,
        head_checkpoint_url=head_checkpoint_url,
        cfg_str=head_config_url,
        head_type=head_type,
        head_scale_count=head_scale_count,
    )
    dataset = LizardDataset(
        local_path="~/projects/lab_hackathon_2024/Bio-Image-Datasets/downloads",
    )
    sample = dataset(111)
    image = sample["image"]
    ground_truth = sample["semantic_mask"]

    segmentation_logits = model(image)
    segmented_image = utils.render_segmentation(segmentation_logits, head_scale_count)

    outdir = "../../outputs"
    os.makedirs(outdir, exist_ok=True)
    plt.imshow(segmented_image)
    plt.savefig(os.path.join(outdir, "segmented_image.png"), bbox_inches="tight")

    # Assuming `segmentation_logits` is the predicted segmentation map and `ground_truth` is the actual label
    score = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    score_value = score(segmentation_logits, ground_truth)
    print("Jaccard Score:", score_value)
