import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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
