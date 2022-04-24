# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import numpy as np
import torch
import torch.nn as nn
from mmdet.models import build_backbone

EPS = 1e-5

def masks2palette(masks):
    """
    Convert binary object masks to palette
    Zero indicates background (no annotations)
    Args:
        masks: np.array, NxHxW, where N indicates number of objects
    """
    palette = np.zeros(masks.shape[1:], dtype=np.int8)
    for i, mask in enumerate(masks):
        palette += mask * (i + 1)
    return palette


def palette2filter(palette, neighbor_sizes=None, bidirection=True):
    """
    Generate indicator function to filter out background in loss.
    Same shape as PA target, with 1 as object-object/bg, 0 as bg-bg.
    Args:
        palette: segmentation palette
        neighbor_sizes: the distance used to compute pairwise relations
        bidirection: if True, generate across all (8) neighbors per distance.
            Otherwise, generate 4 (due to symmetry)
    """

    def generate_cross_filter(number_of_neighbors, dist):
        cross_filter = np.zeros((number_of_neighbors,) + palette.shape, dtype=np.bool)
        cross_filter[0, dist:, :] = np.logical_or(palette[dist:, :], palette[:-dist, :])
        cross_filter[1, :, dist:] = np.logical_or(palette[:, dist:], palette[:, :-dist])
        cross_filter[2, dist:, dist:] = np.logical_or(
            palette[dist:, dist:], palette[:-dist, :-dist]
        )
        cross_filter[3, :-dist, dist:] = np.logical_or(
            palette[:-dist, dist:], palette[dist:, :-dist]
        )
        if number_of_neighbors == 8:
            cross_filter[4, :-dist, :] = np.logical_or(
                palette[:-dist, :], palette[dist:, :]
            )
            cross_filter[5, :, :-dist] = np.logical_or(
                palette[:, dist:], palette[:, :-dist]
            )
            cross_filter[6, :-dist, :-dist] = np.logical_or(
                palette[dist:, dist:], palette[:-dist, :-dist]
            )
            cross_filter[7, dist:, :-dist] = np.logical_or(
                palette[:-dist, dist:], palette[dist:, :-dist]
            )
        return cross_filter

    # manual assign default to avoid list input
    if neighbor_sizes is None:
        neighbor_sizes = [1]
    number_of_span = len(neighbor_sizes)
    number_of_neighbors_per_span = 8 if bidirection else 4
    potential_filter = np.zeros(
        (number_of_neighbors_per_span * number_of_span,) + palette.shape, dtype=np.bool
    )
    for neighbor_idx, dist in enumerate(neighbor_sizes):
        offset = neighbor_idx * number_of_neighbors_per_span
        potential_filter[
            offset : offset + number_of_neighbors_per_span
        ] = generate_cross_filter(number_of_neighbors_per_span, dist)
    return potential_filter


def palette2weight(
    palette, neighbor_sizes=None, bidirection=True, weight_type="uniform"
):
    """
    Use to weight per-pixel loss of PA training (NOT used in paper)
    Weight Type supports:
        1. Uniform weight
        2. Inverse weight
        3. Square root inverse weight (inverse_sqrt)
    """

    def generate_cross_weight(number_of_neighbors, dist):
        cross_filter = np.zeros((number_of_neighbors,) + palette.shape, dtype=np.bool)
        cross_filter[0, dist:, :] = np.maximum(palette[dist:, :], palette[:-dist, :])
        cross_filter[1, :, dist:] = np.maximum(palette[:, dist:], palette[:, :-dist])
        cross_filter[2, dist:, dist:] = np.maximum(
            palette[dist:, dist:], palette[:-dist, :-dist]
        )
        cross_filter[3, :-dist, dist:] = np.maximum(
            palette[:-dist, dist:], palette[dist:, :-dist]
        )
        if number_of_neighbors == 8:
            cross_filter[4, :-dist, :] = np.maximum(
                palette[:-dist, :], palette[dist:, :]
            )
            cross_filter[5, :, :-dist] = np.maximum(
                palette[:, dist:], palette[:, :-dist]
            )
            cross_filter[6, :-dist, :-dist] = np.maximum(
                palette[dist:, dist:], palette[:-dist, :-dist]
            )
            cross_filter[7, dist:, :-dist] = np.maximum(
                palette[:-dist, dist:], palette[dist:, :-dist]
            )
        return cross_filter

    weight = np.ones_like(palette)
    if weight_type != "uniform":
        mask_inds = np.unique(palette)
        for mask_idx in mask_inds:
            binary_mask = palette == mask_idx
            if mask_idx == 0:
                weight[binary_mask] = 0.0
                continue
            mask_cnt = np.sum(palette == mask_idx)
            mask_weight = mask_cnt
            if weight_type == "inverse":
                mask_weight /= 1.0
            elif weight_type == "inverse_sqrt":
                mask_weight = np.sqrt(mask_weight)
                mask_weight /= 1.0
            else:
                print("unsupported loss weight type, use uniform")
                mask_weight = 1.0
            weight[binary_mask] = mask_weight

    if neighbor_sizes is None:
        neighbor_sizes = [1]
    number_of_span = len(neighbor_sizes)
    number_of_neighbors_per_span = 8 if bidirection else 4
    potential_weight = np.zeros(
        (number_of_neighbors_per_span * number_of_span,) + palette.shape, dtype=np.bool
    )
    for neighbor_idx, dist in enumerate(neighbor_sizes):
        offset = neighbor_idx * number_of_neighbors_per_span
        potential_weight[
            offset : offset + number_of_neighbors_per_span
        ] = generate_cross_weight(number_of_neighbors_per_span, dist)
    return potential_weight


def palette2affinity(palette, neighbor_sizes=None, bidirection=True):
    """
    convert palette (HxW, np) to pairwise affinity
    """

    def generate_cross_affinity(number_of_neighbors, dist):
        """
        generate single scale affinity
        """
        cross_affinity = np.zeros((number_of_neighbors,) + palette.shape, dtype=np.int8)
        # top
        cross_affinity[0, dist:, :] = palette[dist:, :] == palette[:-dist, :]
        # left
        cross_affinity[1, :, dist:] = palette[:, dist:] == palette[:, :-dist]
        # left, top
        cross_affinity[2, dist:, dist:] = (
            palette[dist:, dist:] == palette[:-dist, :-dist]
        )
        # left, down
        cross_affinity[3, :-dist, dist:] = (
            palette[:-dist, dist:] == palette[dist:, :-dist]
        )
        if number_of_neighbors == 8:
            cross_affinity[4, :-dist, :] = palette[dist:, :] == palette[:-dist, :]
            cross_affinity[5, :, :-dist] = palette[:, dist:] == palette[:, :-dist]
            cross_affinity[6, :-dist, :-dist] = (
                palette[dist:, dist:] == palette[:-dist, :-dist]
            )
            cross_affinity[7, dist:, :-dist] = (
                palette[:-dist, dist:] == palette[dist:, :-dist]
            )
        return cross_affinity

    # manual assign default to avoid list input
    if neighbor_sizes is None:
        neighbor_sizes = [1]
    number_of_span = len(neighbor_sizes)
    number_of_neighbors_per_span = 8 if bidirection else 4
    pairwise_affinity = np.zeros(
        (number_of_neighbors_per_span * number_of_span,) + palette.shape, dtype=np.int8
    )
    for neighbor_idx, dist in enumerate(neighbor_sizes):
        offset = neighbor_idx * number_of_neighbors_per_span
        pairwise_affinity[
            offset : offset + number_of_neighbors_per_span
        ] = generate_cross_affinity(number_of_neighbors_per_span, dist)
    return pairwise_affinity


def collapse_affinity(affinity, bg_filter, reduction=None):
    """
    Pooling used to convert affinity into a 1-channel boundary probability
    """
    if reduction is None:
        return affinity, bg_filter
    bg_filter = np.max(bg_filter, axis=0, keepdims=True)
    if reduction == "mean":
        affinity = np.mean(affinity.astype(float), axis=0, keepdims=True)
    elif reduction == "min":
        affinity = np.min(affinity, axis=0, keepdims=True)
    elif reduction == "max":
        affinity = np.max(affinity, axis=0, keepdims=True)
    return affinity, bg_filter


def instance_mask2affinity(
    instance_mask, bidirection=True, reduction=None, weight_type="uniform"
):
    masks = instance_mask.to_ndarray()
    palette = masks2palette(masks)
    bg_filter = palette2weight(
        palette, bidirection=bidirection, weight_type=weight_type
    )
    affinity = palette2affinity(palette, bidirection=bidirection)
    affinity, bg_filter = collapse_affinity(affinity, bg_filter, reduction=reduction)
    return torch.from_numpy(affinity), torch.from_numpy(bg_filter)


class PairwiseAffinityHead(nn.Sequential):
    """
    Pairwise affinity predictor
    """

    def __init__(
        self,
        in_channels=2048,
        channels=8,
        norm_type="BN",
    ):
        inter_channels = in_channels // 4
        if norm_type == "BN":
            norm_layer = nn.BatchNorm2d(inter_channels)
        else:
            norm_layer = nn.GroupNorm(32, inter_channels)
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer,
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super(PairwiseAffinityHead, self).__init__(*layers)


class PairwiseAffinityHeadUperNet(nn.Module):
    """
    UperNet version of predictor head; leverage FPN
    Reference: https://github.com/CSAILVision/unifiedparsing
    """

    def __init__(
        self,
        channels=8,
        fc_dim=2048,
        pool_scales=(1, 2, 3, 6),
        fpn_inplanes=(256, 512, 1024, 2048),
        fpn_dim=256,
        norm_type="BN",
        return_feat = True,
    ):
        super(PairwiseAffinityHeadUperNet, self).__init__()

        # helper block
        self.return_feat = return_feat
        def conv3x3_bn_relu(in_planes, out_planes, stride=1, norm_layer="BN"):
            "3x3 convolution + BN + relu"
            if norm_type == "BN":
                norm_layer = nn.BatchNorm2d(out_planes)
            else:
                norm_layer = nn.GroupNorm(32, out_planes)
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                norm_layer,
                nn.ReLU(inplace=True),
            )

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            if norm_type == "BN":
                norm_layer = nn.BatchNorm2d(512)
            else:
                norm_layer = nn.GroupNorm(32, 512)
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    norm_layer,
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(
            fc_dim + len(pool_scales) * 512, fpn_dim, 1
        )

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            if norm_type == "BN":
                norm_layer = nn.BatchNorm2d(fpn_dim)
            else:
                norm_layer = nn.GroupNorm(32, fpn_dim)
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    norm_layer,
                    nn.ReLU(inplace=True),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(
                nn.Sequential(
                    conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
                )
            )
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, channels, kernel_size=1),
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    nn.functional.interpolate(
                        pool_scale(conv5),
                        (input_size[2], input_size[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
            )  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                nn.functional.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fusion_out = torch.cat(fusion_list, 1)
        if self.return_feat:
            x = self.conv_last[0](fusion_out)
            return x
        x = self.conv_last(fusion_out)
        return x

backbone=dict(
    type="ResNet",
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type="BN", requires_grad=True),
    # norm_cfg=dict(type="GN", requires_grad=True, num_groups=32),
    norm_eval=False,
    style="pytorch",
    strides=(1, 2, 2, 2),
)

class PA_module(nn.Module):
    def __init__(self, backbone=backbone,return_feat=True):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.classifier = PairwiseAffinityHeadUperNet(channels=1,return_feat=True)

    def init_weights(self, pretrained="torchvision://resnet50"):
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained)

    def forward(self,img):
        x = self.backbone(img)
        pred_pa = self.classifier(x)
        return pred_pa