
import einops
import torch
import torch.nn as nn

from benchmark.simple_segmentation_model import clean_str, load_model_and_transform
from benchmark.unetr import forward_musk_embed

from .blocks import Interpolate, _make_fusion_block


class DPTSegmentationHead(torch.nn.Module):
    def __init__(self, num_classes,
                 num_features,
                 out_shape=None, in_shape=None,
                 groups=1, use_bn=False,
                 channels_last=False):
        super().__init__()
        if out_shape is None:
            out_shape = [1024, 1024, 1024, 1024]
        if in_shape is None:
            in_shape = [1024, 1024, 1024, 1024]

        self.output_conv = nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(num_features, num_classes, kernel_size=1),
                #Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )

        self.channels_last = channels_last

        self.layer1_rn = nn.Conv2d(
            in_shape[0],
            out_shape[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.layer2_rn = nn.Conv2d(
            in_shape[1],
            out_shape[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.layer3_rn = nn.Conv2d(
            in_shape[2],
            out_shape[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape[3],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.refinenet1 = _make_fusion_block(num_features, use_bn)
        self.refinenet2 = _make_fusion_block(num_features, use_bn)
        self.refinenet3 = _make_fusion_block(num_features, use_bn)
        self.refinenet4 = _make_fusion_block(num_features, use_bn)

    def forward(self, x0, x1, x2, x3):
        """X contains the features from the encoder."""
        layer_1_rn = self.layer1_rn(x0)
        layer_2_rn = self.layer2_rn(x1)
        layer_3_rn = self.layer3_rn(x2)
        layer_4_rn = self.layer4_rn(x3)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)

        out = self.output_conv(path_1)
        return out


class DPT(nn.Module):
    """DPT.

    Args:
        nn (_type_): _description_
    """

    def __init__(
            self,
            model_name,
            num_classes,
            **kwargs
        ):
            # For simplicity, we will assume that extract layers must have a length of 4
            super().__init__()
            self.model, self.transform, model_dim, self.patch_size, self.image_size = load_model_and_transform(
                model_name, features_only=True
            )
            self.model_name = clean_str(model_name)
            if not isinstance(model_dim, int):
                self.embed_dim = model_dim[-1]
            else:
                self.embed_dim = model_dim

            self.head = DPTSegmentationHead(
                num_classes=num_classes,
                num_features=self.embed_dim,
                in_shape=[1024, 1024, 1024, 1024],
                out_shape=[1024, 1024, 1024, 1024],
                groups=1,
                use_bn=True,
                channels_last=True,
            )

    def forward(self, x):
        if self.model_name == "phikonv2":
            x = self.transform(x, return_tensors="pt")["pixel_values"]
            x = x.cuda()
        else:
            x = self.transform(x)

        if self.model_name == "musk":
            outputs = forward_musk_embed(self.model.beit3, visual_tokens=x)
            patch_embeddings = outputs["encoder_states"][-4:]
        else:
            patch_embeddings = self.model(x)  # output shape (b, d, p, p) where b=batch_size, d=hidden_dim, p=patch_size

        if self.model_name == "phikonv2":
            patch_embeddings = patch_embeddings["hidden_states"][-4:]

        if self.model_name in ["phikonv2", "titan", "musk"]:
            reshaped_embed = []
            for layer in patch_embeddings:
                reshaped_embed.append(
                    einops.rearrange(
                        layer[:, 1:, :],
                        "b (p1 p2) d -> b d p1 p2",
                        p1=self.image_size // self.patch_size,
                        p2=self.image_size // self.patch_size,
                    )
                )
            patch_embeddings = reshaped_embed

        # Debug print
        #for layer in patch_embeddings:
        #    print("layer", layer.shape)

        z0, z1, z2, z3 = patch_embeddings
        out = self.head(z0, z1, z2, z3)
        return out
