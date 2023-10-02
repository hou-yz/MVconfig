import numpy as np
import torch


def dist_action(act1, act2, action_names, xy_coef=1.0, yaw_coef=1.0, return_xys=False):
    dist_ = None
    if 'x' in action_names and 'y' in action_names:
        idx = [action_names.index('x'), action_names.index('y')]
        xy1, xy2 = act1[..., idx], act2[..., idx]
        dist_loc_ = dist_l2(xy1, xy2)
        dist_ = dist_ + dist_loc_ * xy_coef if dist_ is not None else dist_loc_ * xy_coef
    if 'yaw' in action_names:
        idx = action_names.index('yaw')
        yaw1, yaw2 = act1[..., idx], act2[..., idx]
        delta_xy1 = torch.stack([torch.cos(torch.deg2rad(yaw1)), torch.sin(torch.deg2rad(yaw1))], dim=-1)
        delta_xy2 = torch.stack([torch.cos(torch.deg2rad(yaw2)), torch.sin(torch.deg2rad(yaw2))], dim=-1)
        dist_rot_ = dist_l2(delta_xy1, delta_xy2)
        dist_ = dist_ + dist_rot_ * yaw_coef if dist_ is not None else dist_rot_ * yaw_coef
    if return_xys:
        return dist_, (xy1, xy2, delta_xy1, delta_xy2)
    return dist_


def dist_l2(xy1, xy2):
    return ((xy1 - xy2) ** 2).sum(-1) ** 0.5


def dist_angle(yaw1, yaw2):
    delta_xy1 = torch.stack([torch.cos(torch.deg2rad(yaw1)), torch.sin(torch.deg2rad(yaw1))], dim=-1)
    delta_xy2 = torch.stack([torch.cos(torch.deg2rad(yaw2)), torch.sin(torch.deg2rad(yaw2))], dim=-1)
    return dist_l2(delta_xy1, delta_xy2)


def tanh_prime(x, thres=20):
    # tanh'(x) = torch.log(4 / (torch.exp(x) + torch.exp(-x)) ** 2))
    output = np.log(4) - 2 * torch.log(torch.exp(x) + torch.exp(-x))
    # For numerical stability the implementation reverts to the linear function when |x| > thres
    output[x > thres] = np.log(4) - 2 * x[x > thres]
    output[x < -thres] = np.log(4) + 2 * x[x < -thres]
    return output


def expectation(probs, x_range, func, n_points=1000, device='cpu'):
    if isinstance(x_range[0], float):
        B, C = 1, 1
        x_range = [torch.ones(B, C) * (x_range[0]), torch.ones(B, C) * (x_range[0])]
    elif len(x_range[0].shape) == 0:
        B, C = 1, 1
        x_range = [x_range[0].reshape([B, C]), x_range[0].reshape([B, C])]
    elif len(x_range[0].shape) == 1:
        B, C = 1, x_range.shape[0]
        x_range = [x_range[0].reshape([B, C]), x_range[0].reshape([B, C])]
    elif len(x_range[0].shape) == 2:
        B, C = x_range[0].shape
        pass
    else:
        raise Exception
    x = torch.empty([B, C, n_points], device=device)
    for b in range(B):
        for c in range(C):
            x[b, c] = torch.linspace(float(x_range[0][b, c]), float(x_range[1][b, c]), n_points, device=device)
    log_probs = torch.stack([probs.log_prob(x[:, :, i]) for i in range(n_points)], dim=2)
    return torch.trapz(torch.exp(log_probs) * func(x), x)


def to_tensor(x, dtype=torch.float, device='cpu'):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype, device=device)
    return x


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat
