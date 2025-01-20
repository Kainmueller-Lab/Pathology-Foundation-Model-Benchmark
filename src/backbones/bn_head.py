# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmengine.registry import MODELS

@MODELS.register_module()
class BNHead(BaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        #print("self.in_channels", self.in_channels, self.channels)
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs, with_bn=False):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        #print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        if with_bn:
            feats = self.bn(x)
        else:
            feats = x
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        features = self._forward_feature(inputs)
        # features shape b, hidden_dim * scales, h, w
        #print("Head output", features.shape)
        hidden_dim = self.channels

        # TODO: single linear layer to match hidden dim
        for i in range(int(features.shape[1]/hidden_dim)):
            features_scale = features[:, i*hidden_dim:(i+1)*hidden_dim, :, :]
            if i == 0:
                output = self.cls_seg(features_scale)
            else:
                output += self.cls_seg(features_scale)

        #print(f'BNHead fwd output.shape {output.shape}')
        return output
