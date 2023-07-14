import numpy as np

import json


# Projective flattening, scales homogeneous coordinates so that last coordinate is always one
def pflat(x):
    if len(x.shape) == 1:
        x /= x[-1]
    else:
        x /= x[-1, :]
    return x


# calculate location L2 distance
def loc_dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def write_config_mv_det(
    path, map, cam_x, cam_y, cam_fov, cam_pos_lst, cam_dir_lst, spawn_count, spawn_area
):
    config = {
        "map": map,
        "cam_x": cam_x,
        "cam_y": cam_y,
        "cam_fov": cam_fov,
        "cam_pos_lst": cam_pos_lst,
        "cam_dir_lst": cam_dir_lst,
        "spawn_count": spawn_count,
        "spawn_area": spawn_area,
    }
    with open(path, "w") as fp:
        json.dump(config, fp, indent=4)


def read_config(path):
    rtn = None
    with open(path, "r") as fp:
        rtn = json.load(fp)
    return rtn


def get_origin(cfg):
    spawn_area = cfg["spawn_area"]
    minx, miny = spawn_area[0], spawn_area[2]
    return minx, miny
