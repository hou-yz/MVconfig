import numpy as np


def get_worldgrid_from_pos(pos, map_width, map_expand):
    grid_x = pos % (map_width * map_expand)
    grid_y = pos // (map_width * map_expand)
    return np.array([grid_x, grid_y], dtype=int)


def get_pos_from_worldgrid(worldgrid, map_width, map_expand):
    grid_x, grid_y = worldgrid
    return grid_x + grid_y * map_width * map_expand


def get_worldgrid_from_worldcoord(world_coord, origin_x, origin_y, map_expand):
    coord_x, coord_y = world_coord
    grid_x = (coord_x - origin_x) * map_expand
    grid_y = (coord_y - origin_y) * map_expand
    return np.array([grid_x, grid_y], dtype=int)


def get_worldcoord_from_worldgrid(worldgrid, origin_x, origin_y, map_expand):
    grid_x, grid_y = worldgrid
    coord_x = origin_x + grid_x / map_expand
    coord_y = origin_y + grid_y / map_expand
    return np.array([coord_x, coord_y])


def get_worldcoord_from_pos(pos, origin_x, origin_y, map_width, map_expand):
    grid = get_worldgrid_from_pos(pos, map_width, map_expand)
    return get_worldcoord_from_worldgrid(grid, origin_x, origin_y, map_expand)


def get_pos_from_worldcoord(world_coord, origin_x, origin_y, map_width, map_expand):
    grid = get_worldgrid_from_worldcoord(world_coord, origin_x, origin_y, map_expand)
    return get_pos_from_worldgrid(grid, map_width, map_expand)
