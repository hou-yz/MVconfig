import numpy as np
import torch

from src.utils.tensor_utils import to_tensor


# def build_camera_matrices(folder: Path, output_K=False):
#     txt_path = folder / "cameras.json"
#     text = txt_path.read_text()
#     cams_obj = json.loads(text)
#     f = cams_obj["instrinsics"]["f"]
#     Cx = cams_obj["instrinsics"]["Cx"]
#     Cy = cams_obj["instrinsics"]["Cy"]

#     cameras = dict()
#     for cam in cams_obj["cams"]:
#         values = {"f": f, "Cx": Cx, "Cy": Cy}
#         for key in ("x", "y", "z", "pitch", "roll", "yaw"):
#             values[key] = cam[key]

#         cam_id = int(cam["id"])
#         P, K = build_cam(values)
#         cameras[cam_id] = P

#     if output_K:
#         return cameras, K

#     return cameras

def action2proj_mat(dataset, act, cam):
    action = dataset.base.env.action(act, cam)
    [x, y, z, pitch, yaw, roll, fov] = action

    image_h, image_w = dataset.img_shape
    f = image_w / (2.0 * torch.tan(torch.deg2rad(fov) / 2))
    Cx = image_w / 2.0
    Cy = image_h / 2.0

    _, K, Rt = build_cam(x, y, z, pitch, yaw, roll, f, Cx, Cy)

    proj_mat = dataset.get_world_imgs_trans(K, Rt)

    return proj_mat


def build_cam(x, y, z, pitch, yaw, roll, f, Cx, Cy):
    is_float = isinstance(x, float)
    device = x.device if isinstance(x, torch.Tensor) else 'cpu'
    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    flip = torch.zeros([3, 3], device=device)
    flip[0, 1] = 1
    flip[1, 2] = -1
    flip[2, 0] = 1

    # intrinsic
    K = torch.zeros([3, 3], device=device)
    K[0, 0] = f
    K[0, 2] = Cx
    K[1, 1] = f
    K[1, 2] = Cy
    K[2, 2] = 1

    # Let t_c be a column vector describing the location of the camera-center in world coordinates, and let R_c
    #  be the rotation matrix describing the camera's orientation with respect to the world coordinate axes

    # Translation
    t_c = torch.zeros([3], device=device)
    t_c[0] = x
    t_c[1] = y
    t_c[2] = z

    # Rotation
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # https://github.com/carla-simulator/carla/issues/2516#issuecomment-861761770
    # Carla performs negative roll, then negative pitch, then yaw.
    pitch, yaw, roll = to_tensor(pitch), to_tensor(yaw), to_tensor(roll)
    pitch_ = -torch.deg2rad(pitch)
    R_pitch = torch.zeros([3, 3], device=device)
    R_pitch[0, 0] = torch.cos(pitch_)
    R_pitch[0, 2] = torch.sin(pitch_)
    R_pitch[1, 1] = 1
    R_pitch[2, 0] = -torch.sin(pitch_)
    R_pitch[2, 2] = torch.cos(pitch_)
    yaw_ = torch.deg2rad(yaw)
    R_yaw = torch.zeros([3, 3], device=device)
    R_yaw[0, 0] = torch.cos(yaw_)
    R_yaw[0, 1] = -torch.sin(yaw_)
    R_yaw[1, 0] = torch.sin(yaw_)
    R_yaw[1, 1] = torch.cos(yaw_)
    R_yaw[2, 2] = 1
    roll_ = -torch.deg2rad(roll)
    R_roll = torch.zeros([3, 3], device=device)
    R_roll[0, 0] = 1
    R_roll[1, 1] = torch.cos(roll_)
    R_roll[1, 2] = -torch.sin(roll_)
    R_roll[2, 1] = torch.sin(roll_)
    R_roll[2, 2] = torch.cos(roll_)
    R_c = R_yaw @ R_pitch @ R_roll
    # The transformation matrix that describes the camera's pose is then [R_c|t_c].
    Rt_c = torch.zeros([4, 4], device=device)
    Rt_c[:3, :3] = R_c
    Rt_c[:3, 3] = t_c
    Rt_c[3, 3] = 1
    # Then the extrinsic matrix is obtained by inverting the camera's pose matrix:
    # https://ksimek.github.io/2012/08/22/extrinsic/
    Rt = flip @ torch.inverse(Rt_c)[:3]
    P = K @ Rt
    if is_float:
        P, K, Rt = P.numpy(), K.numpy(), Rt.numpy()

    return P, K, Rt


def build_cam_from_configs(cam_config, scene_config):
    # build the camera matrix from the cam and scene configs
    # the cam config the a dictioanry with 7 elements (with fov)
    Cx = scene_config["image_x"] / 2.0
    Cy = scene_config["image_y"] / 2.0
    f = scene_config["image_x"] / (2.0 * np.tan(cam_config["fov"] * np.pi / 360))
    cam_values = cam_config.copy()
    del cam_values["fov"]
    cam_values["Cx"] = Cx
    cam_values["Cy"] = Cy
    cam_values["f"] = f
    return build_cam(**cam_values)


def euler_angles(phi, theta, psi):
    sin = np.sin
    cos = np.cos
    R = [
        [
            cos(theta) * cos(psi),
            -cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi),
            sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi),
        ],
        [
            cos(theta) * sin(psi),
            cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi),
            -sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi),
        ],
        [-sin(theta), sin(phi) * cos(theta), cos(phi) * cos(theta)],
    ]
    return np.array(R, dtype=np.float32)


def is_visible(x, y, z, P, im_h=720, im_w=1280):
    point = np.array([x, y, z, 1.0], dtype=np.float32).reshape((4, 1))
    proj = (P @ point).flatten()
    # Check if in front of the camera
    if proj[-1] > 0.0:
        px, py = proj[:2] / proj[-1]
        if px >= 0 and px <= im_w and py >= 0 and py <= im_h:
            return True

    return False


def is_obj_visible(x, y, z, height, P, im_h=720, im_w=1280, min_height=10.0):
    point = np.array([x, y, z, 1.0], dtype=np.float32).reshape((4, 1))
    proj = (P @ point).flatten()

    # Check if behind the camera
    if proj[-1] < 0.0:
        return False

    px, py = proj[:2] / proj[-1]
    if px >= 0 and px <= im_w and py >= 0 and py <= im_h:
        delta_height = np.array([0, 0, height, 0], dtype=np.float32).reshape((4, 1))
        new_point = point + delta_height
        proj2 = P @ new_point
        proj2 = proj2 / proj2[-1]
        py2 = proj2[1]

        pixel_height = abs(py2 - py)
        if pixel_height >= min_height:
            return True

    return False
