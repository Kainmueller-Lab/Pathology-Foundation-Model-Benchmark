from omegaconf import OmegaConf
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoder
from transformers import AutoImageProcessor
import torch
from benchmark.simple_segmentation_model import clean_str, load_model_and_transform
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from huggingface_hub import login
from dataclasses import dataclass

from simple_segmentation_model import HF_TOKEN, MockModel, clean_str, get_model_dim

# TODO: put into model configs
model_configs = {
    "uni": {
        "channels": [1024, 1024, 1024],
    }
}


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
    login(token=HF_TOKEN)
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
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
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
        model_dim = 1024
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model.eval(), transform, model_dim


@dataclass
class ModelOutput:
    feature_maps: torch.Tensor


class ModelWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

    def forward(self, x):
        # use transform function
        return ModelOutput(self.backbone(x))


class Mask2FormerModel(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model_name = model_name
        self.backbone, self.transform, model_dim = load_model_and_transform(model_name)
        self.backbone = ModelWrapper(self.backbone)
        self.num_classes = num_classes
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
        model_config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
        self.model = Mask2FormerForUniversalSegmentation(model_config)
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
            logits: Output logits of shape (b, num_classes, h, w)
        """
        b, c, h, w = x.shape
        if clean_str(self.model_name) == "phikonv2":
            x = self.transform(x, return_tensors="pt")["pixel_values"]
        else:
            x = self.transform(x)

        out = self.model(x)

        return self.image_processor.post_process_semantic_segmentation(out, target_sizes=[image.size()[::-1]])
