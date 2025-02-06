import torch
import timm
from omegaconf import OmegaConf
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from timm.layers import SwiGLUPacked
from timm.data.transforms import MaybeToTensor
from torchvision import transforms

import musk
from timm.models import create_model
from timm.data.constants import (
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

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
        super(SimpleSegmentationModel, self).__init__()
        self.model_name = model_name
        self.model, self.transform, model_dim = load_model_and_transform(model_name)
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
        Returns:
            logits: Output logits of shape (b, num_classes, h, w)
        """
        b, c, h, w = x.shape
        if clean_str(self.model_name) == "phikonv2":
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
        super(MockModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.model = torch.nn.Conv2d(3, embedding_dim, kernel_size=3)
        self.model.train()

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (b, c, h, w)
        Returns:
            logits: Output logits of shape (b, embedding_dim, h, w)
        """
        return self.model(x)


def load_model_and_transform(model_name):
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
    if clean_str(model_name) == "provgigapath":
        login(token=HF_TOKEN)
        model_cfg = OmegaConf.load("configs/models/provgigapath.yaml")
        model = timm.create_model(f"hf_hub:{model_cfg.url}", pretrained=model_cfg.pretrained)
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
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "phikonv2":
        login(token=HF_TOKEN)
        model_cfg = OmegaConf.load("configs/models/phikonv2.yaml")
        model = AutoModel.from_pretrained(model_cfg.url, trust_remote_code=True)
        model.forward_patches = (
            lambda x: model(x.to(model.device))
            .last_hidden_state[:, 1:, :]
            .reshape(
                x.shape[0],
                int(model_cfg.img_size / model_cfg.patch_size),
                int(model_cfg.img_size / model_cfg.patch_size),
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        transform = AutoImageProcessor.from_pretrained(model_cfg.url)
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "virchow2":
        login(token=HF_TOKEN)
        model_cfg = OmegaConf.load("configs/models/virchow2.yaml")
        model = timm.create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=model_cfg.pretrained,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
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
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "uni":
        login(token=HF_TOKEN)
        model_cfg = OmegaConf.load("configs/models/uni.yaml")
        model = timm.create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=model_cfg.pretrained,
            init_values=model_cfg.init_values,
            dynamic_img_size=model_cfg.dynamic_img_size,
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
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "uni2":
        login(token=HF_TOKEN)
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
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "titan":
        login(token=HF_TOKEN)
        model_cfg = OmegaConf.load("configs/models/titan.yaml")
        titan = AutoModel.from_pretrained(model_cfg.url, trust_remote_code=True)
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
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "musk":
        model_cfg = OmegaConf.load("configs/models/musk.yaml")
        model = create_model("musk_large_patch16_384")
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
        model_dim = get_model_dim(model, img_size=model_cfg.img_size)
    elif clean_str(model_name) == "mock":
        model = MockModel(1024)
        transform = lambda x: x
        model.forward_patches = lambda x: model(x)
        model_dim = 1024
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model, transform, model_dim


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
    with torch.no_grad():
        model_dim = model.forward_patches(torch.zeros(1, 3, img_size, img_size)).shape[1]
    return model_dim
