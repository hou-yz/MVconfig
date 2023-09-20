import numpy as np
import torch
from src.utils.tensor_utils import to_tensor


def project_2d_points(project_mat, input_points, check_visible=False):
    device = input_points.device if isinstance(input_points, torch.Tensor) else 'cpu'
    project_mat, input_points = to_tensor(project_mat), to_tensor(input_points)
    if input_points.dim() == 1: input_points = input_points[None]
    # N points, each C=2 dimensions
    N, C = input_points.shape
    input_points = torch.cat([input_points, torch.ones([N, 1], device=device)], dim=1)
    output_points_ = project_mat @ input_points.T
    output_points = output_points_[:2, :] / output_points_[2, :]
    if check_visible:
        is_visible = output_points_[2, :] > 0
        return output_points.T, is_visible
    else:
        return output_points.T


def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat, z=0):
    project_mat = get_worldcoord_from_imgcoord_mat(intrinsic_mat, extrinsic_mat, z)
    return project_2d_points(project_mat, image_coord)


def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat, z=0):
    project_mat = get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z)
    return project_2d_points(project_mat, world_coord)


def get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    """image of shape C,H,W; xy indexging; x,y (w,h)
    world of shape N_row, N_col; xy indexging; z in meters by default
    """
    device = intrinsic_mat.device if isinstance(intrinsic_mat, torch.Tensor) else 'cpu'
    intrinsic_mat, extrinsic_mat = to_tensor(intrinsic_mat), to_tensor(extrinsic_mat)
    threeD2twoD = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, z], [0, 0, 1]], dtype=torch.float, device=device)
    project_mat = intrinsic_mat @ extrinsic_mat @ threeD2twoD
    return project_mat


def get_worldcoord_from_imgcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    """image of shape C,H,W; xy indexging; x,y (w,h)
    world of shape N_row, N_col; xy indexging; z in meters by default
    """
    project_mat = torch.inverse(get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z))
    return project_mat
