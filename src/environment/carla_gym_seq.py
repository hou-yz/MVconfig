import copy
import os
from queue import Queue
import random
import subprocess
import time

import carla
import cv2
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

from src.environment.cameras import build_cam
from src.environment.utils import loc_dist, pflat

# render quality of CARLA
QUALITY = ("Epic", "Low")
# Hack-import objects for spawning
SpawnActor = carla.command.SpawnActor


def run_carla(carla_path, off_screen=False, quality="Epic", gpu=2):
    assert quality in QUALITY
    script_path = os.path.join(carla_path, "CarlaUE4.sh")
    prompt = f"bash {script_path}"
    if off_screen:
        prompt += " -RenderOffScreen"
    prompt += f" -quality-level={quality}"
    if not os.path.exists(script_path):
        raise FileNotFoundError("CarlaUE4.sh file not found")
    prompt += f" -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter={gpu}"
    game_proc = subprocess.Popen(prompt, shell=True)
    # wait for carla to start
    time.sleep(5.0)
    # One can use game_proc.poll() to check server status
    # None -> running, otherwise -> exited
    return game_proc


def encode_camera_cfg(cfg, opts):
    [x, y, z, pitch, yaw, roll, fov] = cfg
    # convert cfg: location/rotation/fov from different ranges to [-1, 1]
    # location
    # x, y, z \in [x_min, x_max], [y_min, y_max], [z_min, z_max]
    x_min, x_max, y_min, y_max, z_min, z_max = opts["spawn_area"]
    x = (x - x_min) / (x_max - x_min) * 2 - 1
    y = (y - y_min) / (y_max - y_min) * 2 - 1
    z = (z - z_min) / (z_max - z_min) * 2 - 1
    location = np.array([x, y, z])
    # rotation
    # pitch, yaw, roll \in [-90, 90], [-180, 180], [-180, 180]
    pitch = pitch / 90
    yaw = yaw / 180
    roll = roll / 180
    rotation = np.array([pitch, yaw, roll])
    # fov \in [0, 180]
    fov = np.array(fov / 90 - 1).reshape(-1)
    # put location, rotation, fov together
    cfg = np.concatenate([location, rotation, fov])
    return cfg


def decode_camera_cfg(cfg, opts):
    # action is for a SINGLE camera
    # convert cfg: location/rotation/fov from [-1, 1] to different ranges
    [x, y, z, pitch, yaw, roll, fov] = cfg
    # location
    # x, y, z \in [x_min, x_max], [y_min, y_max], [z_min, z_max]
    x_min, x_max, y_min, y_max, z_min, z_max = opts["spawn_area"]
    x = (x + 1) / 2 * (x_max - x_min) + x_min
    y = (y + 1) / 2 * (y_max - y_min) + y_min
    z = (z + 1) / 2 * (z_max - z_min) + z_min
    location = np.array([x, y, z])
    # rotation
    # pitch, yaw, roll \in [-90, 90], [-180, 180], [-180, 180]
    pitch = pitch * 90
    yaw = yaw * 180
    roll = roll * 180
    rotation = np.array([pitch, yaw, roll])
    # fov \in [0, 180]
    fov = np.array((fov + 1) * 90).reshape(-1)
    # put location, rotation, fov together
    cfg = np.concatenate([location, rotation, fov])
    return cfg


def draw_bbox(obs, info):
    imgs = obs["images"]
    gts = info["pedestrian_gts"]
    fig, axs = plt.subplots(2, 2, figsize=(39, 22))

    for cam, img in imgs.items():
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for anno in gts:
            anno = anno["views"][cam]
            bbox = tuple(
                [
                    int(anno["xmin"]),
                    int(anno["ymin"]),
                    int(anno["xmax"]),
                    int(anno["ymax"]),
                ]
            )
            if bbox[0] == -1 and bbox[1] == -1 and bbox[2] == -1 and bbox[3] == -1:
                continue
            img = cv2.rectangle(img, bbox[:2], bbox[2:], (0, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[cam // 2, cam % 2].imshow(img)
        axs[cam // 2, cam % 2].set_title(f'Camera {cam + 1}')

    return fig, axs


class CarlaCameraSeqEnv(gym.Env):
    """
    The CARLA environment for single-camera multi-frame pedestrian detection
    """

    def __init__(self, opts, seed=None, host="127.0.0.1", port=2000):
        self.opts = opts

        # if seed is provided, set seed to generators
        # otherwise randomly initialise generators
        if seed is not None:
            self.random_generator = random.Random(seed)
            self.np_random_generator = np.random.default_rng(seed)
        else:
            self.random_generator = random.Random()
            self.np_random_generator = np.random.default_rng()

        # Connect to the CARLA simulator
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.opts["map"])

        # CarlaX is xy indexing; x,y (w,h) (n_col,n_row)
        # x_min, x_max, _, _, _, _ = opts["spawn_area"]
        # self.map_width = x_max - x_min
        # print(self.map_width)

        self.num_cam = self.opts["num_cam"]
        # step counter for deciding whether episode ends
        self.step_counter = 0

        # Define your environment's observation space
        self.observation_space = spaces.Dict(
            {
                "images": spaces.Box(
                    low=0,
                    high=255,
                    shape=(int(self.opts['cam_y']), int(self.opts['cam_x']), 3),
                    dtype=np.uint8,
                ),
                "camera_configs": spaces.Box(-1, 1, shape=(7,)),
                "step": spaces.Box(-1, self.num_cam - 1, shape=(1,), dtype=int)
            }
        )

        # Define your environment's action space
        self.action_space = spaces.Box(-1, 1, shape=(len(self.opts["env_action_space"].split("-")),))

        # Define any other attributes or variables needed for your environment
        # turn on sync mode
        traffic_manager = self.client.get_trafficmanager(8000)
        settings = self.world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # world actors
        self.camera_configs = {}
        self.camera_intrinsics = {}
        self.camera_extrinsics = {}
        self.cameras = {}
        self.img_cam_buffer = {}
        self.pedestrians = []
        self.pedestrian_gts = []

    def action(self, act):
        # camera config for the next camera
        # allow more flexible choice of action space (x-y-z-pitch-yaw-roll-fov)
        # convert normalised action space to unnormalised ones
        action_space = self.opts["env_action_space"].split("-")
        action_dim_dict = {'x': 0, 'y': 1, 'z': 2, 'pitch': 3, 'yaw': 4, 'roll': 5, 'fov': 6}
        _action = np.zeros(self.observation_space['camera_configs'].shape)
        for i, action_name in enumerate(action_space):
            _action[action_dim_dict[action_name]] = act[i]
        _action = np.clip(_action, -1, 1)
        _cfg = decode_camera_cfg(_action, self.opts)
        _location, _rotation, _fov = _cfg[:3], _cfg[3:6], _cfg[6]
        # default settings for limited action space
        location, rotation, fov = np.array(self.opts["cam_pos_lst"])[self.step_counter], \
            np.array(self.opts["cam_dir_lst"])[self.step_counter], \
            np.array(self.opts["cam_fov"]).reshape([1])
        if 'x' in action_space: location[0] = _location[0]
        if 'y' in action_space: location[1] = _location[1]
        if 'z' in action_space: location[2] = _location[2]
        if 'pitch' in action_space: rotation[0] = _rotation[0]
        if 'yaw' in action_space: rotation[1] = _rotation[1]
        if 'roll' in action_space: rotation[2] = _rotation[2]
        if 'fov' in action_space: fov = _fov

        action = np.concatenate([location, rotation, fov], axis=0)
        return action

    def reset(self, seed=None):
        # if a new seed is provided, set generator to used new seed
        # otherwise use old seed
        if seed is not None:
            self.random_generator = random.Random(seed)
            self.np_random_generator = np.random.default_rng(seed)

        # Reset the environment to its initial state and return the initial observation
        self.reset_cameras()
        self.respawn_pedestrians()
        self.step_counter = 0

        # NOTE: render all cameras by default
        observation = {
            "images": self.render(),
            "camera_configs": {cam: encode_camera_cfg(self.camera_configs[cam], self.opts)
                               for cam in range(self.num_cam)},
            "step": self.step_counter
        }
        info = {"pedestrian_gts": self.pedestrian_gts,
                "camera_intrinsics": self.camera_intrinsics,
                "camera_extrinsics": self.camera_extrinsics}  # Set any additional information

        # NOTE: Remember that Python only returns a reference to these objects
        # you may need to use copy.deepcopy() to avoid effects from further steps
        return observation, info

    def step(self, action):
        self.step_counter += 1
        # the input action would be an array of 7 numbers, as defined in action space
        # values are in the range of 0-1
        # Perform one step in the environment based on the given action
        action = self.action(action)
        cfg = np.stack(self.camera_configs.values())
        cfg[self.step_counter] = action
        self.reset_cameras(cfg)

        # update pedestrian bbox from each camera view
        for i, pedestrian in enumerate(self.pedestrians):
            actor = self.world.get_actor(pedestrian)
            self.pedestrian_gts[i]["views"][self.step_counter] = self.get_pedestrian_view(actor, self.step_counter)
            # self.pedestrian_gts[i]["views"] = {cam: self.get_pedestrian_view(actor, cam) for cam in range(self.num_cam)}

        # Update the state, calculate the reward, and check for termination
        # Set the current observation
        observation = {
            "images": self.render(),
            "camera_configs": {cam: encode_camera_cfg(self.camera_configs[cam], self.opts)
                               for cam in range(self.num_cam)},
            "step": self.step_counter
        }
        # Set the reward for the current step
        reward = 0
        # Set condition for the end of episode: after a fixed number of step() call
        done = self.step_counter + 1 >= self.num_cam  # Set whether the episode has terminated or not
        # Set any additional information, the info can be used to calculate reward outside gym env
        info = {"pedestrian_gts": copy.deepcopy(self.pedestrian_gts),
                "camera_intrinsics": self.camera_intrinsics,
                "camera_extrinsics": self.camera_extrinsics, }

        # NOTE: Remember that Python only returns a reference to these objects
        # you may need to use copy.deepcopy() to avoid effects from further steps
        return observation, reward, done, info

    def render(self):
        # Render the environment
        images = {}
        # start listening the camera images
        for cam, camera in self.cameras.items():
            camera.listen(self.img_cam_buffer[cam].put)
        self.world.tick()
        # wait for sync until get all images
        for cam, queue_buffer in self.img_cam_buffer.items():
            image = queue_buffer.get()
            images[cam] = process_img(image)
        # end listening
        for camera in self.cameras.values():
            camera.stop()
        return images

    def close(self):
        # Clean up any resources or connections
        # after capturing all frames, destroy all actors
        self.client.apply_batch(
            [carla.command.DestroyActor(id) for id in self.pedestrians]
        )
        for camera in self.cameras.values():
            camera.destroy()

    def respawn_pedestrians(self):
        # Destroy existing actors, create new ones randomly
        self.client.apply_batch(
            [carla.command.DestroyActor(id) for id in self.pedestrians]
        )
        # spawn parameter, make the spawn area 0.5m smaller
        min_x, max_x = self.opts["spawn_area"][0:2]
        min_y, max_y = self.opts["spawn_area"][2:4]
        min_x, min_y = min_x + 0.5, min_y + 0.5
        max_x, max_y = max_x - 0.5, max_y - 0.5
        # Spawn pedestrians
        spawn_points = []
        for _ in range(self.opts["spawn_count"]):
            spawn_point = carla.Transform()
            loc = None
            while loc is None:
                # random initialise x, y, z
                loc = carla.Location()
                loc.x = self.random_generator.uniform(min_x, max_x)
                loc.y = self.random_generator.uniform(min_y, max_y)
                # avoid collision with ground
                loc.z = 1.0
                if len(spawn_points):
                    distances = [
                        loc_dist(previous.location, loc) for previous in spawn_points
                    ]
                    if min(distances) < 1.0:
                        # To precent collisions with other road users
                        loc = None
            spawn_point.location = loc
            # randomise yaw -> [0, 359] -> [0, 360)
            spawn_point.rotation = carla.Rotation(0, self.random_generator.random() * 360, 0)
            spawn_points.append(spawn_point)
        batch = []
        bps_pedestrians = self.world.get_blueprint_library().filter(
            "walker.pedestrian.*"
        )
        for spawn_point in spawn_points:
            walker_bp = self.random_generator.choice(bps_pedestrians)
            # make sure all pedestrians are vincible
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            batch.append(SpawnActor(walker_bp, spawn_point))
        # apply spawn pedestrian
        results = self.client.apply_batch_sync(batch, True)
        self.pedestrians = []
        for res in results:
            if not res.error:
                # if error happens, very likely to be spawning failure caused by collision
                self.pedestrians.append(res.actor_id)
        # print(f"{len(self.pedestrians)} pedestrians spawned")

        # print("Stabilizing world...")
        # for _ in range(50):
        #     self.world.tick()
        #     time.sleep(0.01)
        self.world.tick()
        # print("World stabilized")

        self.update_pedestrian_gts()

    def update_pedestrian_gts(self):
        self.pedestrian_gts = []
        for pedestrian in self.pedestrians:
            # 1. new ground truth format
            actor = self.world.get_actor(pedestrian)
            loc = actor.get_location()

            bbox = actor.bounding_box
            l = bbox.extent.x * 2
            w = bbox.extent.y * 2
            h = bbox.extent.z * 2
            # Pedestrians' z value is at their middle height for some reason
            ped_z = loc.z - h / 2.0
            v = actor.get_velocity()
            rot = actor.get_transform().rotation
            forward = rot.get_forward_vector()

            # ped_worldcoord = [loc.x, loc.y]
            # ped_pos = int(get_pos_from_worldcoord(ped_worldcoord, *self.origin, self.map_width, self.opts["map_expand"]))

            self.pedestrian_gts.append(
                {
                    "id": pedestrian,
                    "x": loc.x,
                    "y": loc.y,
                    "z": ped_z,
                    "l": l,
                    "w": w,
                    "h": h,
                    "v_x": v.x,
                    "v_y": v.y,
                    "v_z": v.z,
                    "pitch": rot.pitch,
                    "roll": rot.roll,
                    "yaw": rot.yaw,
                    "forward_x": forward.x,
                    "forward_y": forward.y,
                    "forward_z": forward.z,
                    # "positionID": ped_pos,
                    "views": {cam: self.get_pedestrian_view(actor, cam) for cam in range(self.num_cam)},
                }
            )
        # print(f"Collected {len(self.pedestrian_gts)} pedestrian information")

    def get_pedestrian_view(self, actor, cam):
        bbox = actor.bounding_box
        verts = bbox.get_world_vertices(actor.get_transform())

        # prepare 2D bbox
        x_max = 0
        x_min = float(self.opts["cam_x"])
        y_max = 0
        y_min = float(self.opts["cam_y"])
        for vert in verts:
            # convert vert to homogeneous coordinate, vert is a carla.Location
            vert_homo = np.array([vert.x, vert.y, vert.z, 1])
            p_homo = self.camera_intrinsics[cam] @ self.camera_extrinsics[cam] @ vert_homo
            p = pflat(p_homo)
            if p[0] > x_max:
                x_max = p[0]
            if p[0] < x_min:
                x_min = p[0]
            if p[1] > y_max:
                y_max = p[1]
            if p[1] < y_min:
                y_min = p[1]

        if (
                x_max > 100
                and x_min < float(self.opts["cam_x"]) - 100
                and y_max > 100
                and y_min < float(self.opts["cam_y"]) - 100
        ):
            pedestrian_view = {"viewNum": cam, "xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max, }
        else:
            # same as the convention (MultiviewX), feed invalid value (-1) if bbox outside picture
            pedestrian_view = {"viewNum": cam, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1, }

        return pedestrian_view

    def reset_cameras(self, cfg=None):
        # destroy existing cameras
        for camera in self.cameras.values():
            camera.destroy()
        self.camera_configs = {}
        self.camera_intrinsics = {}
        self.camera_extrinsics = {}
        self.cameras = {}
        self.img_cam_buffer = {}

        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.opts["cam_x"]))
        camera_bp.set_attribute("image_size_y", str(self.opts["cam_y"]))

        if cfg is None:
            locations = np.array(self.opts["cam_pos_lst"])
            rotations = np.array(self.opts["cam_dir_lst"])
            fovs = float(self.opts["cam_fov"]) * np.ones(self.num_cam)
        else:
            locations, rotations, fovs = cfg[:, :3], cfg[:, 3:6], cfg[:, 6]

        for cam, (cam_pos, cam_dir, fov) in enumerate(zip(locations, rotations, fovs)):
            camera_bp.set_attribute("fov", str(fov))
            loc = carla.Location(*cam_pos)
            rot = carla.Rotation(*cam_dir)
            camera_init_trans = carla.Transform(loc, rot)
            # spawn the camera
            camera = self.world.spawn_actor(camera_bp, camera_init_trans)
            # record camera related information
            self.cameras[cam] = camera

        # wait for one tick to update the camera actors
        self.world.tick()

        for cam, camera in self.cameras.items():
            # save camera configs, rather than projection matrices
            # projection/intrinsic/extrinsic matrices can be built from configs
            cam_config, intrinsic, extrinsic = get_camera_config(camera)
            self.camera_configs[cam] = cam_config
            self.camera_intrinsics[cam] = intrinsic
            self.camera_extrinsics[cam] = extrinsic
            self.img_cam_buffer[cam] = Queue(maxsize=0)


def process_img(img):
    # img is carla.libcarla.Image object;
    # returns a ndarray
    img_bgra = np.reshape(np.copy(img.raw_data), (img.height, img.width, 4))
    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
    return img_rgb


def get_camera_config(camera):
    image_w = int(camera.attributes["image_size_x"])
    image_h = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])

    transform = camera.get_transform()
    loc = transform.location
    rot = transform.rotation

    f = image_w / (2.0 * np.tan(fov * np.pi / 360))
    Cx = image_w / 2.0
    Cy = image_h / 2.0

    # camera value is consist of 9 elements
    cam_value = {
        "x": loc.x,
        "y": loc.y,
        "z": loc.z,
        "pitch": rot.pitch,
        "roll": rot.roll,
        "yaw": rot.yaw,
        "f": f,
        "Cx": Cx,
        "Cy": Cy,
    }

    # config is consist of 7 elements [x, y, z, pitch, yaw, roll, fov]
    cam_config = [loc.x, loc.y, loc.z, rot.pitch, rot.yaw, rot.roll, fov, ]

    _, intrinsic, extrinsic = build_cam(**cam_value)
    return cam_config, intrinsic, extrinsic


if __name__ == '__main__':
    import json
    from tqdm import tqdm

    with open('cfg/RL/2.cfg', "r") as fp:
        dataset_config = json.load(fp)

    env = CarlaCameraSeqEnv(dataset_config)
    for i in tqdm(range(4000)):
        observation, info = env.reset()
        done = False
        while not done:
            observation, reward, done, info = env.step(np.random.rand(7))
