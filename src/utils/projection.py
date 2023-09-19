import numpy as np
import torch
from src.utils.tensor_utils import to_tensor


def project_2d_points(project_mat, input_points):
    device = input_points.device if isinstance(input_points, torch.Tensor) else 'cpu'
    project_mat, input_points = to_tensor(project_mat), to_tensor(input_points)
    if input_points.dim() == 1: input_points = input_points[None]
    # N points, each C=2 dimensions
    N, C = input_points.shape
    input_points = torch.cat([input_points, torch.ones([N, 1], device=device)], dim=1)
    output_points = project_mat @ input_points.T
    output_points = output_points[:2, :] / output_points[2, :]
    return output_points.T


def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat, z=0):
    project_mat = get_worldcoord_from_imgcoord_mat(intrinsic_mat, extrinsic_mat, z)
    return project_2d_points(project_mat, image_coord)


def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat, z=0):
    project_mat = get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z)
    return project_2d_points(project_mat, world_coord)


def get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    device = intrinsic_mat.device if isinstance(intrinsic_mat, torch.Tensor) else 'cpu'
    """image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
    world of shape N_row, N_col; indexed as specified in the dataset attribute (xy or ij)
    z in meters by default
    """
    intrinsic_mat, extrinsic_mat = to_tensor(intrinsic_mat), to_tensor(extrinsic_mat)
    threeD2twoD = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, z], [0, 0, 1]], dtype=torch.float, device=device)
    project_mat = intrinsic_mat @ extrinsic_mat @ threeD2twoD
    return project_mat


def get_worldcoord_from_imgcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    """image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
    world of shape N_row, N_col; indexed as specified in the dataset attribute (xy or ij)
    z in meters by default
    """
    project_mat = torch.inverse(get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z))
    return project_mat
