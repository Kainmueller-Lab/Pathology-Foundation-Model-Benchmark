import torch
import timm
import os
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from huggingface_hub import login
from dotenv import load_dotenv
import numpy as np
from pathlib import Path


# load the environment variables
dotenv_path = Path(__file__).parents[2] / ".env"
load_dotenv(dotenv_path=dotenv_path)
HF_TOKEN = os.getenv("HF_TOKEN")


class SimpleSegmentationModel(torch.nn.Module):
    """Simple segmentation model that uses a backbone and a linear head."""
    def __init__(self, model_name, num_classes):
        super(SimpleSegmentationModel, self).__init__()
        self.model, self.transform, model_dim = load_model_and_transform(model_name)
        self.head = torch.nn.Conv2d(
            in_channels=model_dim, out_channels=num_classes, kernel_size=1
        )
        self.num_classes = num_classes
        self.model.eval()
        self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze the backbone of the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the backbone of the model."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (b, c, h, w)
        Returns:
            logits: Output logits of shape (b, num_classes, h, w)
        """
        b,c,h,w = x.shape
        x = self.transform(x)
        patch_embeddings = self.model(x) # 1, 1024, 14, 14
        logits= self.head(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=(h,w), mode="bilinear", align_corners=False)
        return logits


def load_model_and_transform(model_name):
    if model_name == "UNI":
        login(token=HF_TOKEN)
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True
        )
        model.forward = lambda x: model.forward_features(x)[:,1:,:].reshape(x.shape[0], 14,14, -1).permute(0,3,1,2)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        with torch.no_grad():
            model_dim = model(torch.zeros(1,3,224,224)).shape[1]
    elif model_name == "XYZ":
        NotImplementedError("Model not implemented")
    return model, transform, model_dim
