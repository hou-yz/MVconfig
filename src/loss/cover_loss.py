import numpy as np
import torch
import torch.nn.functional as F
from src.utils.tensor_utils import expectation, tanh_prime, dist_action, dist_l2
from src.utils.projection import project_2d_points
from src.loss.centernet_loss import _transpose_and_gather_feat

LINSPACE_STEPS = 200


def cover_loss(proj_mats, covermap, history_cover_maps, b_world_gts, dataset, cover_clamp):
    device = covermap.device
    B, _, H, W = covermap.shape
    # image of shape C,H,W (C,N_row,N_col); xy-indexing; x,y (w,h) (n_col,n_row)
    # image border, xy-indexing
    h, w = dataset.img_shape
    img_top = torch.stack([torch.linspace(0.2 * w, 0.8 * w, LINSPACE_STEPS),
                           torch.linspace(0.2 * h, 0.2 * h, LINSPACE_STEPS)], dim=1)
    img_left = torch.stack([torch.linspace(0.2 * w, 0.2 * w, LINSPACE_STEPS),
                            torch.linspace(0.2 * h, 0.8 * h, LINSPACE_STEPS)], dim=1)
    img_bottom = torch.stack([torch.linspace(0.2 * w, 0.8 * w, LINSPACE_STEPS),
                              torch.linspace(0.8 * h, 0.8 * h, LINSPACE_STEPS)], dim=1)
    img_right = torch.stack([torch.linspace(0.8 * w, 0.8 * w, LINSPACE_STEPS),
                             torch.linspace(0.2 * h, 0.8 * h, LINSPACE_STEPS)], dim=1)
    image_points = torch.stack([img_top, img_left, img_bottom, img_right]).flatten(0, 1).to(device)
    proj_points = torch.stack([project_2d_points(proj_mats[b], image_points, check_visible=True)[0] for b in range(B)])
    proj_top, proj_left, proj_bottom, proj_right = proj_points.unflatten(1, [4, LINSPACE_STEPS]).permute([1, 0, 2, 3])
    # image center
    img_center = torch.tensor([[w / 2, h / 2]], device=device)
    proj_center = torch.stack([project_2d_points(proj_mats[b], img_center, check_visible=True)[0] for b in range(B)])
    # pedestrian location, xy-indexing
    pedestrian_idx = b_world_gts['idx'].to(device)
    pedestrian_xys = torch.stack([pedestrian_idx % W, pedestrian_idx // W], dim=-1)
    # distances: pedestrian location to borders & center
    dist_top, dist_left, dist_bottom, dist_right, dist_center = [
        # ((border[:, None] - pedestrian_xys[:, :, None]) ** 2).min(dim=2)[0].sum(-1) ** 0.5
        torch.cdist(pedestrian_xys.float(), border).min(dim=2)[0]
        for border in [proj_top, proj_left, proj_bottom, proj_right, proj_center]]
    # visibility and weight for pedestrian in each view
    is_visible = _transpose_and_gather_feat(covermap.detach().bool(), pedestrian_idx)[..., 0]
    map_weights = torch.tanh(1 + history_cover_maps.sum(dim=1, keepdims=True)) - \
                  torch.tanh(history_cover_maps.sum(dim=1, keepdims=True))
    # map_weights = ~history_cover_maps.sum(dim=1, keepdims=True).bool()
    pedestrian_weights = _transpose_and_gather_feat(map_weights, pedestrian_idx)[..., 0]
    # loss: pull the <nearest> border closer to invisible pedestrian
    loss_ = torch.stack([dist_top, dist_left, dist_bottom, dist_right]).min(dim=0)[0] * ~is_visible
    loss = (torch.clamp(loss_, 0, cover_clamp[1]) * pedestrian_weights)[b_world_gts['reg_mask']].mean()
    return loss
