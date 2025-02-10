from omegaconf import OmegaConf
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoder
from transformers import AutoImageProcessor
import torch
from benchmark import utils
from benchmark.simple_segmentation_model import clean_str
import timm
import musk
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from huggingface_hub import login
from dataclasses import dataclass
from timm.layers import SwiGLUPacked
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
from timm.data.transforms import MaybeToTensor
from timm.data.constants import (
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from transformers import MaskFormerImageProcessor
from benchmark.simple_segmentation_model import HF_TOKEN, MockModel, clean_str, get_model_dim
import numpy as np


def get_model_dim(model, img_size):
    """
    Get the dimensionality of the features output by the model.

    Parameters
    ----------
    model : nn.Module
        The model to get the feature dimensionality from.

    Returns
    -------
    int
        The dimensionality of the features output by the model.
    """
    model_dims = []
    with torch.no_grad():
        model_ouput = model(torch.zeros(1, 3, img_size, img_size))
        if hasattr(model_ouput, "hidden_states"):
            hidden_states = model_ouput.hidden_states
        else:
            hidden_states = model_ouput

        for layer_ in hidden_states:
            print("layer", layer_.shape)
            if len(layer_.shape) > 1:
                dim_idx = 1
            else:
                dim_idx = 0
            model_dims.append(layer_.shape[dim_idx])

    return model_dims


def load_m2f_model_and_transform(model_name):
    """
    Load the specified model and a transform object to prepare input data according to the model's
    requirements.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model (nn.Module): The pre-trained model.
        transform (callable): A transform object to prepare input data according to the model's
            requirements.
        model_dim (int): The dimensionality of the model's output features.
    """
    login(token=HF_TOKEN)
    if clean_str(model_name) == "provgigapath":
        model_cfg = OmegaConf.load("configs/models/provgigapath.yaml")
        model = timm.create_model(f"hf_hub:{model_cfg.url}", pretrained=model_cfg.pretrained, features_only=True)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "phikonv2":
        model_cfg = OmegaConf.load("configs/models/phikonv2.yaml")
        model = AutoModel.from_pretrained(model_cfg.url, trust_remote_code=True, output_hidden_states=True)
        transform = AutoImageProcessor.from_pretrained(model_cfg.url)
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "virchow2":
        model_cfg = OmegaConf.load("configs/models/virchow2.yaml")
        model = timm.create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=model_cfg.pretrained,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            features_only=True,
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "uni":
        model_cfg = OmegaConf.load("configs/models/uni.yaml")
        model = timm.create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=model_cfg.pretrained,
            init_values=model_cfg.init_values,
            dynamic_img_size=model_cfg.dynamic_img_size,
            features_only=True,
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "uni2":
        model_cfg = OmegaConf.load("configs/models/uni2.yaml")
        model = timm.create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=model_cfg.pretrained,
            img_size=model_cfg.img_size,
            patch_size=model_cfg.patch_size,
            depth=model_cfg.depth,
            num_heads=model_cfg.num_heads,
            init_values=model_cfg.init_values,
            embed_dim=model_cfg.embed_dim,
            mlp_ratio=model_cfg.mlp_ratio,
            num_classes=model_cfg.num_classes,
            no_embed_class=model_cfg.no_embed_class,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=model_cfg.reg_tokens,
            dynamic_img_size=model_cfg.dynamic_img_size,
            features_only=True,
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "titan":
        model_cfg = OmegaConf.load("configs/models/titan.yaml")
        titan = AutoModel.from_pretrained(model_cfg.url, trust_remote_code=True, output_hidden_states=True)
        model, _ = titan.return_conch()
        transform = transforms.Compose(
            [
                transforms.Resize(model_cfg.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                MaybeToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "musk":
        model_cfg = OmegaConf.load("configs/models/musk.yaml")
        model = timm.models.create_model("musk_large_patch16_384")
        musk.utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
        model.to(device="cuda", dtype=torch.float32)
        transform = transforms.Compose(
            [
                transforms.Resize(model_cfg.img_size, interpolation=3, antialias=True),
                transforms.CenterCrop((model_cfg.img_size, model_cfg.img_size)),
                MaybeToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ]
        )

        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "mock":
        model = MockModel(1024)
        transform = lambda x: x
        model_dim = 1024
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    if hasattr(model_cfg, "channels"):
        model_dim = model_cfg.channels
    print("model_dim", model_dim)
    return model.eval(), transform, model_dim


@dataclass
class ModelOutput:
    feature_maps: torch.Tensor


class ModelWrapper(torch.nn.Module):
    def __init__(self, backbone, model_transform):
        super().__init__()

        self.backbone = backbone
        self.model_transform = model_transform

    def forward(self, x):
        # Transform should not be used because m2f already has an image processor
        # x = self.model_transform(x)
        return ModelOutput(self.backbone(x))


class Mask2FormerModel(torch.nn.Module):

    def __init__(self, model_name, num_classes, img_shape=(224, 224)):
        super().__init__()
        self.model_name = model_name
        self.backbone, self.transform, model_dim = load_m2f_model_and_transform(model_name)
        self.backbone = ModelWrapper(self.backbone, self.transform)
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.image_processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-tiny-cityscapes-semantic",
            use_fast=True,
            do_rescale=False,
            size=img_shape,
            image_mean=IMAGENET_DEFAULT_MEAN,
            image_std=IMAGENET_DEFAULT_STD,
            num_labels=num_classes - 1, # exclude background
        )

        #model_config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-base-IN21k-ade-semantic")
        model_config = Mask2FormerConfig.from_pretrained('/fast/AG_Kainmueller/nkoreub/Pathology-Foundation-Model-Benchmark/configs/mask2former_lizard.json')
        model_config.num_labels = self.num_classes - 1 # because m2f adds weight for background class during loss computation

        self.model = Mask2FormerForUniversalSegmentation(model_config)

        #print('MODEL CHILDREN')
        #for name, child in self.model.named_children():
        #    print(name, child)

        # --> reveals: class_predictor Linear(in_features=256, out_features=7, bias=True)

        self.model.model.pixel_level_module.encoder = self.backbone
        self.model.model.pixel_level_module.encoder.channels = model_dim
        self.model.model.pixel_level_module.decoder = Mask2FormerPixelDecoder(model_config, feature_channels=model_dim)
        self.model.eval()

    def freeze_model(self):
        """Freeze the model."""
        for param in self.model.model.pixel_level_module.encoder.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        """Unfreeze the model."""
        for param in self.model.model.pixel_level_module.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (b, c, h, w)
        Returns:
            logits: Output logits of shape (b, h, w)
        """

        # Uncomment if you want to use the model's transforms
        # if clean_str(self.model_name) == "phikonv2":
        #    x = self.transform(x, return_tensors="pt")["pixel_values"]
        # else:
        #    x = self.transform(x)

        out = self.model(**x)
        batch_size = out.class_queries_logits.shape[0]
        print('out.class_queries_logits requires_grad', out.class_queries_logits.requires_grad) # True
        print('out.mask_queries_logits requires_grad', out.masks_queries_logits.requires_grad) # True

        semantic_seg = self.image_processor.post_process_semantic_segmentation(out, target_sizes=[self.img_shape for _ in range(batch_size)])
        return semantic_seg
