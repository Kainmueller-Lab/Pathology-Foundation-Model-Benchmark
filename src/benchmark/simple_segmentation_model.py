import torch
import timm
from torchvision import transforms
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from timm.layers import SwiGLUPacked
from huggingface_hub import login

from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

# load the environment variables
dotenv_path = Path(__file__).parents[2] / ".env"
load_dotenv(dotenv_path=dotenv_path)
HF_TOKEN = os.getenv("HF_TOKEN")

model_urls = {
    "provgigapath": "prov-gigapath/prov-gigapath",
    "phikonv2": "owkin/phikon-v2",
    "virchow2": "paige-ai/Virchow2",
    "uni": "MahmoodLab/UNI",
    "titan": "MahmoodLab/TITAN"
}

class SimpleSegmentationModel(torch.nn.Module):
    """Simple segmentation model that uses a backbone and a linear head."""
    def __init__(self, model_name, num_classes):
        """
        Initialize the SimpleSegmentationModel with the specified model and number of classes.

        Args:
            model_name (str): The name of the model to load, one of Prov-GigaPath, Phikon-v2, Virchow2, UNI, Titan.
            num_classes (int): The number of output classes for the segmentation task.
        """
        super().init()
        self.model, self.transform, model_dim = load_model_and_transform(model_name)
        self.head = torch.nn.Conv2d(
            in_channels=model_dim, out_channels=num_classes, kernel_size=1
        )
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
        x = self.transform(x)
        patch_embeddings = self.model(x) # output shape (b, d, p, p) where b=batch_size, d=hidden_dim, p=patch_size
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
    if clean_str(model_name) == "provgigapath":
        model = timm.create_model(f"hf_hub:{model_urls['provgigapath']}", pretrained=True)
        model.forward = lambda x: model.forward_features(x)[:, 1:, :].reshape(x.shape[0], 14, 14, -1).permute(0,3,1,2)
        transform =  create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        get_model_dim(model)
    elif clean_str(model_name) == "phikonv2":
        model = AutoModel.from_pretrained(model_urls['phikonv2'])
        model.forward = lambda x: model(x).last_hidden_state[:, 1:, :].reshape(x.shape[0], 14, 14, -1).permute(0,3,1,2)
        transform = AutoImageProcessor.from_pretrained(model_urls['phikonv2'])
        get_model_dim(model)
    elif clean_str(model_name) == "virchow2":
        model = timm.create_model(f"hf-hub:{model_urls['virchow2']}", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        # start from index 5 of last_hidden_state since first token is CLS and token 1-4 are registers
        model.forward = lambda x: model(x)[:, 5:, :].reshape(x.shape[0], 16, 16, -1).permute(0,3,1,2)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        get_model_dim(model)
    elif clean_str(model_name) == "uni":
        login()
        model = timm.create_model(f"hf-hub:{model_urls['uni']}", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        model.forward = lambda x: model.forward_features(x)[:, 1:, :].reshape(x.shape[0], 14, 14, -1).permute(0,3,1,2)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        get_model_dim(model)
    elif clean_str(model_name) == "titan":
        model = AutoModel.from_pretrained(model_urls['titan'], trust_remote_code=True)
        conch, transform = model.return_conch()
        model.forward = lambda x: conch._modules['trunk'].forward_features(x)[:,1:,:].reshape(x.shape[0], 14, 14, -1).permute(0,3,1,2)
        get_model_dim(model)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model, transform, model_dim


def clean_str(string):
    return string.replace(" ", "").replace("-", "").lower()

def get_model_dim(model):
    with torch.no_grad():
        model_dim = model(torch.zeros(1, 3, 224, 224)).shape[-1]
    return model_dim


    