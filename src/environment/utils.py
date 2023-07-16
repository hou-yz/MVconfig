# Projective flattening, scales homogeneous coordinates so that last coordinate is always one
def pflat(x):
    if len(x.shape) == 1:
        x /= x[-1]
    else:
        x /= x[-1, :]
    return x


# calculate location L2 distance
def loc_dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
