import torch
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from huggingface_hub import login


class SimpleSegmentationModel(torch.Module):
    """Simple segmentation model that uses a backbone and a linear head."""
    def __init__(self, model_name, num_classes):
        self.model, self.transform, model_dim = load_model_and_transform(model_name)
        self.head = torch.nn.Linear(model_dim, num_classes, requires_grad=True)
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
        patch_embeddings = self.model(x) # 1, 14, 14, 1024
        logits= self.head(patch_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=(h,w), mode="bilinear", align_corners=False
        )
        return logits


def load_model_and_transform(model_name):
    if model_name == "UNI":
        login()
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True
        )
        model.forward = lambda x: model.forward_features(x)[:,1:,:].reshape(x.shape[0], 14,14, -1)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        with torch.no_grad():
            model_dim = model(torch.zeros(1,3,224,224)).shape[-1]
    elif model_name == "XYZ":
        NotImplementedError("Model not implemented")
    return model, transform, model_dim
