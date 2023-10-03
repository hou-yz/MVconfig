import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.parameters import *



def aggregate_feat(feat, aggregation='mean'):
    overall_feat = cover_mean(feat) if aggregation == 'mean' else feat.max(dim=1)[0]
    return overall_feat


def cover_mean(feat):
    B, N, C, H, W = feat.shape
    feat = feat.permute([0, 3, 4, 1, 2]).flatten(0, 2)  # B,H,W,N,C -> L,N,C
    cover_mask = feat.norm(dim=-1) != 0  # L,N
    feat_mean = feat.sum(dim=1) / (cover_mask.sum(dim=1)[..., None] + 1e-8)  # L,C
    return feat_mean.unflatten(0, [B, H, W]).permute([0, 3, 1, 2])


def cover_mean_std(feat):
    B, N, C, H, W = feat.shape
    feat = feat.permute([0, 3, 4, 1, 2]).flatten(0, 2)  # B,H,W,N,C -> L,N,C
    cover_mask = feat.norm(dim=-1) != 0  # L,N
    feat_mean = feat.sum(dim=1) / (cover_mask.sum(dim=1)[..., None] + 1e-8)  # L,C
    feat_std = torch.zeros_like(feat_mean)  # L,C
    idx = cover_mask.sum(dim=1) > 1  # L
    # pad std for locations covered by zero or one camera
    feat_std[~idx] = STD_PADDING_VALUE
    # covered by more than one camera
    feat_std[idx] = (((feat[idx] - feat_mean[idx][:, None]) ** 2 * cover_mask[idx][..., None]).sum(dim=1) /
                     cover_mask[idx].sum(dim=1)[:, None] + 1e-8) ** 0.5
    return (feat_mean.unflatten(0, [B, H, W]).permute([0, 3, 1, 2]),
            feat_std.unflatten(0, [B, H, W]).permute([0, 3, 1, 2]))


class MultiviewBase(nn.Module):
    def __init__(self, dataset, aggregation='max'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.aggregation = aggregation
        self.control_module = None

    def forward(self, imgs, M=None, proj_mats=None, visualize=False):
        if self.control_module is None:
            feat, aux_res = self.get_feat(imgs, M, proj_mats, visualize)
            overall_res = self.get_output(feat, visualize)
            return overall_res, aux_res
        else:
            raise NotImplementedError

    def get_feat(self, imgs, M, proj_mats, visualize=False):
        raise NotImplementedError

    def get_output(self, feat, visualize=False):
        raise NotImplementedError
