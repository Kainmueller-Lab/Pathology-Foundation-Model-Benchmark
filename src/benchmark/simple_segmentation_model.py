import os
from pathlib import Path

import timm
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from musk import utils
from omegaconf import OmegaConf
from timm.data import resolve_data_config
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.data.transforms import MaybeToTensor
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

# load the environment variables
dotenv_path = Path(__file__).parents[2] / ".env"
load_dotenv(dotenv_path=dotenv_path)
HF_TOKEN = os.getenv("HF_TOKEN")

train_cfg = OmegaConf.load("configs/config.yaml")


class SimpleSegmentationModel(torch.nn.Module):
    """Simple segmentation model that uses a backbone and a linear head."""

    def __init__(self, model_name, num_classes):
        """
        Initialize the SimpleSegmentationModel with the specified model and number of classes.

        Args:
            model_name (str): The name of the model to load, one of Prov-GigaPath, Phikon-v2,
                Virchow2, UNI, Titan.
            num_classes (int): The number of output classes for the segmentation task.
        """
        super().__init__()
        self.model_name = clean_str(model_name)
        self.model, self.transform, model_dim, _ = load_model_and_transform(model_name)
        self.head = torch.nn.Conv2d(in_channels=model_dim, out_channels=num_classes, kernel_size=1)
        self.model.eval()
        self.freeze_model()

    def freeze_model(self):
        """Freeze the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        """Unfreeze the model."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (b, c, h, w)

        Returns
        -------
            logits: Output logits of shape (b, num_classes, h, w)
        """
        b, c, h, w = x.shape
        if self.model_name == "phikonv2":
            x = self.transform(x, return_tensors="pt")["pixel_values"]
        else:
            x = self.transform(x)
        patch_embeddings = self.model.forward_patches(
            x
        )  # output shape (b, d, p, p) where b=batch_size, d=hidden_dim, p=patch_size
        logits = self.head(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


class MockModel(torch.nn.Module):
    """Mock model for testing.

    Args:
        embedding_dim (int): The dimension of the embeddings.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model = torch.nn.Conv2d(3, embedding_dim, kernel_size=3)
        self.model.train()

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (b, c, h, w)

        Returns
        -------
            logits: Output logits of shape (b, embedding_dim, h, w)
        """
        return self.model(x)


def load_model_and_transform(
    model_name: str, features_only: bool = False
) -> tuple[torch.nn.Module, torch.nn.Module, int | list[int]]:
    """Load model and transform to prepare data.

    Args:
        model_name: The name of the model to load.
        features_only: Whether to load the model with features only.

    Returns
    -------
        model: The pre-trained model.
        transform: A transform object to prepare input data according to the model's
            requirements.
        model_dim: The dimensionality of the model's output features.
    """
    login(token=HF_TOKEN)
    model_name = clean_str(model_name)
    if features_only:
        # take features of last 4 layers
        out_indices = (-4, -3, -2, -1)
    else:
        out_indices = None
    if model_name == "provgigapath":
        model_cfg = OmegaConf.load("configs/models/provgigapath.yaml")
        model = timm.create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=model_cfg.pretrained,
            features_only=features_only,
            out_indices=out_indices,
        )
        model.forward_patches = (
            lambda x: model.forward_features(x)[:, 1:, :]
            .reshape(
                x.shape[0],
                int(model_cfg.img_size / model_cfg.patch_size),
                int(model_cfg.img_size / model_cfg.patch_size),
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif model_name == "phikonv2":
        model_cfg = OmegaConf.load("configs/models/phikonv2.yaml")
        model = AutoModel.from_pretrained(model_cfg.url, trust_remote_code=True, output_hidden_states=features_only)
        if not features_only:
            model.forward_patches = (
                lambda x: model(x)
                .last_hidden_state[:, 1:, :]
                .reshape(
                    x.shape[0],
                    int(model_cfg.img_size / model_cfg.patch_size),
                    int(model_cfg.img_size / model_cfg.patch_size),
                    -1,
                )
                .permute(0, 3, 1, 2)
            )
        else:
            # TODO reshape here?
            model.forward_patches = lambda x: model(x).hidden_states[:-4]

        transform = AutoImageProcessor.from_pretrained(model_cfg.url, use_fast=True, do_rescale=False)
    elif model_name == "virchow2":
        model_cfg = OmegaConf.load("configs/models/virchow2.yaml")
        model = timm.create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=model_cfg.pretrained,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            features_only=features_only,
            out_indices=out_indices,
        )
        # start from index 5 of last_hidden_state since first token is CLS, token 1-4 are registers
        model.forward_patches = (
            lambda x: model(x)[:, 5:, :]
            .reshape(
                x.shape[0],
                int(model_cfg.img_size / model_cfg.patch_size),
                int(model_cfg.img_size / model_cfg.patch_size),
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif model_name == "uni":
        model_cfg = OmegaConf.load("configs/models/uni.yaml")
        model = timm.create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=model_cfg.pretrained,
            init_values=model_cfg.init_values,
            dynamic_img_size=model_cfg.dynamic_img_size,
            features_only=features_only,
            out_indices=out_indices,
        )
        model.forward_patches = (
            lambda x: model.forward_features(x)[:, 1:, :]
            .reshape(
                x.shape[0],
                int(model_cfg.img_size / model_cfg.patch_size),
                int(model_cfg.img_size / model_cfg.patch_size),
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif model_name == "uni2":
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
            features_only=features_only,
            out_indices=out_indices,
        )
        model.forward_patches = (
            lambda x: model.forward_features(x)[:, model_cfg.reg_tokens + 1 :, :]
            .reshape(
                x.shape[0],
                int(model_cfg.img_size / model_cfg.patch_size),
                int(model_cfg.img_size / model_cfg.patch_size),
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif model_name == "titan":
        model_cfg = OmegaConf.load("configs/models/titan.yaml")
        titan = AutoModel.from_pretrained(model_cfg.url, trust_remote_code=True, output_hidden_states=features_only)
        model, _ = titan.return_conch()
        transform = transforms.Compose(
            [
                transforms.Resize(model_cfg.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                MaybeToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        model.forward_patches = (
            lambda x: model._modules["trunk"]
            .forward_features(x)[:, 1:, :]
            .reshape(
                x.shape[0],
                int(model_cfg.img_size / model_cfg.patch_size),
                int(model_cfg.img_size / model_cfg.patch_size),
                -1,
            )
            .permute(0, 3, 1, 2)
        )
    elif model_name == "musk":
        model_cfg = OmegaConf.load("configs/models/musk.yaml")
        model = timm.create_model(
            "musk_large_patch16_384",
            features_only=features_only,
            out_indices=out_indices,
        )
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
        # model.to(dtype=torch.float32) # is this necessary?
        transform = transforms.Compose(
            [
                transforms.Resize(model_cfg.img_size, interpolation=3, antialias=True),
                transforms.CenterCrop((model_cfg.img_size, model_cfg.img_size)),
                MaybeToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ]
        )

        model.forward_patches = (
            lambda x: model(
                x.to(device="cuda", dtype=torch.float32),
                with_head=False,
                out_norm=False,
                ms_aug=False,
                return_global=False,
            )[0][:, 1:, :]
            .reshape(
                x.shape[0],
                int(model_cfg.img_size / model_cfg.patch_size),
                int(model_cfg.img_size / model_cfg.patch_size),
                -1,
            )
            .permute(0, 3, 1, 2)
        )
    elif model_name == "mock":
        model = MockModel(1024)
        transform = torch.nn.Identity()
        model.forward_patches = lambda x: model(x)
        model_dim = 1024
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    if hasattr(model_cfg, "channels"):
        model_dim = model_cfg.channels
    else:
        model_dim = get_model_dim(model, img_size=model_cfg.img_size, features_only=features_only)

    return model, transform, model_dim, model_cfg.patch_size


def clean_str(string):
    """
    Clean a string by removing all spaces and hyphens and converting to lower case.

    Parameters
    ----------
    string : str
        The string to clean.

    Returns
    -------
    str
        The cleaned string.
    """
    return string.replace(" ", "").replace("-", "").lower()


def get_model_dim(model, img_size, features_only=False):
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
    mock_input = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        if not features_only:
            model_dim = model.forward_patches(mock_input).shape[1]
        else:
            # in that case model_dim is a list
            if hasattr(model, "feature_info"):
                model_dim = model.feature_info.channels()
            else:
                out_list = model(mock_input)
                model_dim = []
                try:
                    for out in out_list[:4]:
                        if out is not None:
                            if len(out.shape) > 1:
                                dim_idx = 1
                            else:
                                dim_idx = 0
                            model_dim.append(out.shape[dim_idx])
                    while len(model_dim) < 4:
                        model_dim.append(model_dim[-1])

                except ValueError as e:
                    print(e)

    return model_dim
