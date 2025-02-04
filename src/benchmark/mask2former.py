from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoder
from transformers import AutoImageProcessor
import torch
from benchmark.simple_segmentation_model import clean_str, model_urls, load_model_and_transform
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from huggingface_hub import login
from dataclasses import dataclass

from simple_segmentation_model import HF_TOKEN, clean_str

backbone, transform, model_dim = load_model_and_transform("uni")

image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
model_config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")

model_urls = {
    "provgigapath": "prov-gigapath/prov-gigapath",
    "phikonv2": "owkin/phikon-v2",
    "virchow2": "paige-ai/Virchow2",
    "uni": "MahmoodLab/UNI",
    "uni2": "MahmoodLab/UNI2-h",
    "titan": "MahmoodLab/TITAN",
}

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
        model (nn.Module): The pre-trained model that returns a list of features maps like this
            [torch.Tensor [B, C, PatchW, PatchH], torch.Tensor [B, C, PatchW, PatchH], ...]
        transform (callable): A transform object to prepare input data according to the model's
            requirements.
        model_dim list(int): The dimensionality of the model's output features i.e. [1024,1024,1024],
        length of the list should be equal the lenght of the model's output feature maps.
    """

    if model_name in model_urls.keys():
        login(token=HF_TOKEN)
    if clean_str(model_name) == "uni":
        model = timm.create_model(
            f"hf-hub:{model_urls['uni']}", pretrained=True, init_values=1e-5, dynamic_img_size=True, features_only=True
        )
        model.eval()
        model_dim = model_configs[model_name]["channels"]
        transform = None
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform, model_dim


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


wrapped_backbone = ModelWrapper(backbone)

model = Mask2FormerForUniversalSegmentation(model_config)
model.model.pixel_level_module.encoder = wrapped_backbone
model.model.pixel_level_module.encoder.channels = model_configs["uni"]["channels"]
model.model.pixel_level_module.decoder = Mask2FormerPixelDecoder(
    model_config, feature_channels=model_configs["uni"]["channels"]
)
# make mock image batch
image = torch.rand(2, 3, 224, 224)

with torch.no_grad():
    # out = model(image)
    out = backbone(image)
results = image_processor.post_process_semantic_segmentation(out, target_sizes=[image.size()[::-1]])


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
