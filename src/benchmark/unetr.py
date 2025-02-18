# adapted from CellViT
# https://github.com/TIO-IKIM/CellViT-plus-plus/blob/main/cellvit/models/cell_segmentation/cellvit.py
from collections import OrderedDict

import einops
import torch
import torch.nn as nn

from benchmark.simple_segmentation_model import clean_str, load_model_and_transform


class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout.

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # noqa: D102
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout.

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


def forward_musk_embed(
    self,
    visual_tokens=None,
    attn_mask=None,
    vision_masked_position=None,
    incremental_state=None,
    positions=None,
):
    x = self.vision_embed(visual_tokens, vision_masked_position)
    encoder_padding_mask = None
    multiway_split_position = -1

    encoder_out = self.encoder(
        src_tokens=None,
        encoder_padding_mask=encoder_padding_mask,
        attn_mask=attn_mask,
        token_embeddings=x,
        multiway_split_position=multiway_split_position,
        incremental_state=incremental_state,
        positions=positions,
        return_all_hiddens=True,
    )
    return encoder_out


class UnetR(nn.Module):
    """CellViT Modell for cell segmentation.

    U-Net like network with vision transformer as backbone encoder.
    Skip connections are shared between branches, but each network has a distinct encoder

    The modell is having multiple branches:
        * tissue_types: Tissue prediction based on global class token
        * nuclei_binary_map: Binary nuclei prediction
        * hv_map: HV-prediction to separate isolated instances
        * nuclei_type_map: Nuclei instance-prediction
        * [Optional, if regression loss]:
        * regression_map: Regression map for binary prediction

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        embed_dim (int): Embedding dimension of backbone ViT
        input_channels (int): Number of input channels
        depth (int): Depth of the backbone ViT
        num_heads (int): Number of heads of the backbone ViT
        extract_layers: (List[int]): List of Transformer Blocks whose outputs should be returned in addition to the tokens. First blocks starts with 1, and maximum is N=depth.
            Is used for skip connections. At least 4 skip connections needs to be returned.
        mlp_ratio (float, optional): MLP ratio for hidden MLP dimension of backbone ViT. Defaults to 4.
        qkv_bias (bool, optional): If bias should be used for query (q), key (k), and value (v) in backbone ViT. Defaults to True.
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
    """

    def __init__(
        self,
        model_name,
        num_classes,
        drop_rate: float = 0,
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
        self.drop_rate = drop_rate

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3

        self.nuclei_type_maps_decoder = self.create_upsampling_branch(num_classes)

        # self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.tensor_split:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (b, c, h, w)

        Returns
        -------
            logits: Output logits of shape (b, num_classes, h, w)
        """
        # Transform
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
        # for layer in patch_embeddings:
        #    print("layer", layer.shape)

        z0, z1, z2, z3, z4 = x, *patch_embeddings

        logits = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder)
        # maybe resize when patch size is 14?
        return logits

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch.

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns
        -------
            torch.Tensor: Branch Output
        """
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        if b0.shape != b1.shape:  # happens when patch size is 14, then b1 is 256x256 and b0 is 224x224
            _, _, h, w = b0.shape
            b1 = nn.functional.interpolate(b1, size=(h, w), mode="bilinear", align_corners=False)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create Upsampling branch.

        Args:
            num_classes (int): Number of output classes

        Returns
        -------
            nn.Module: Upsampling path
        """
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder

    def freeze_model(self):
        """Freeze the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        """Unfreeze the model."""
        for param in self.model.parameters():
            param.requires_grad = True
