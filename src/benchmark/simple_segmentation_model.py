import torch
import timm
from torchvision import transforms
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from huggingface_hub import login

from transformers import AutoImageProcessor, AutoModel

class SimpleSegmentationModel(torch.nn.Module):
    """Simple segmentation model that uses a backbone and a linear head."""
    def __init__(self, model_name, num_classes):
        super(SimpleSegmentationModel, self).__init__()
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
    """
    Load the specified model and a transform object to prepare input data according to the model's requirements.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model (nn.Module): The pre-trained model.
        transform (callable): A transform object to prepare input data according to the model's requirements.
        model_dim (int): The dimensionality of the model's output features.
    """
    if model_name == "Prov-GigaPath":
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        model.forward = lambda x: model.forward_features(x)[:, 1:, :].reshape(x.shape[0], 14, 14, -1)
        transform =  create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        with torch.no_grad():
            model_dim = model(torch.zeros(1, 3, 224, 224)).shape[-1]
    elif model_name == "Phikon-v2":
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        model.forward = lambda x: model(x).last_hidden_state[:, 1:, :].reshape(x.shape[0], 14, 14, -1)
        transform = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        with torch.no_grad():
            model_dim = model(torch.zeros(1, 3, 224, 224)).shape[-1]
    # uncomment next lines for Virchow2 as soon as we are granted access to the huggingface model
    # elif model_name == "Virchow2":
    #     model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    #     # start from index 5 of last_hidden_state since first token is CLS and token 1-4 are registers
    #     model.forward = lambda x: model(x).last_hidden_state[:, 5:, :].reshape(x.shape[0], 16, 16, -1)
    #     transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    #     with torch.no_grad():
    #         model_dim = model(torch.zeros(1, 3, 224, 224)).shape[-1]
    elif model_name == "UNI":
        login()
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True
        )
        model.forward = lambda x: model.forward_features(x)[:, 1:, :].reshape(x.shape[0], 14, 14, -1)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        with torch.no_grad():
            model_dim = model(torch.zeros(1, 3, 224, 224)).shape[-1]
    elif model_name == "Titan":
        model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        conch, transform = model.return_conch()
        model.forward = lambda x: conch._modules['trunk'].forward_features(x)[:,1:,:].reshape(x.shape[0], 14, 14, -1)
        with torch.no_grad():
            model_dim = conch(torch.zeros(1, 3, 224, 224)).shape[-1]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model, transform, model_dim