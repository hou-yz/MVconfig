import copy
import os
from queue import Queue
import random
import subprocess
import time
import math
import docker
import carla
import cv2
import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.parameters import *
from src.environment.cameras import build_cam


# docker run --privileged --gpus 1 --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh
# docker run --privileged --gpus 1 --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -RenderOffScreen
def docker_run_carla(gpu=0, carla_port=2000):
    client = docker.from_env()
    container = client.containers.run("carlasim/carla:0.9.14",
                                      command=f'/bin/bash ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port={carla_port}',
                                      detach=True,
                                      privileged=True,
                                      network_mode="host",
                                      environment={},
                                      volumes={'/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'rw'}},
                                      device_requests=[docker.types.DeviceRequest(driver='nvidia',
                                                                                  device_ids=[str(gpu)],
                                                                                  capabilities=[['gpu']])])
    # wait for carla to start
    while container.status == "created":
        container.reload()
        time.sleep(2)
    time.sleep(15)
    return container


#  ./CarlaUE4.sh -RenderOffScreen  -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=0 -carla-rpc-port=2000
def run_carla(carla_path, off_screen=False, quality="Epic", gpu=2, port=2000):
    assert quality in QUALITY
    script_path = os.path.join(carla_path, "CarlaUE4.sh")
    prompt = f"bash {script_path}"
    if off_screen:
        prompt += " -RenderOffScreen"
    prompt += f" -quality-level={quality}"
    if not os.path.exists(script_path):
        raise FileNotFoundError("CarlaUE4.sh file not found")
    prompt += f" -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter={gpu} -carla-rpc-port={port}"
    game_proc = subprocess.Popen(prompt, shell=True)
    # wait for carla to start
    time.sleep(5.0)
    # One can use game_proc.poll() to check server status
    # None -> running, otherwise -> exited
    return game_proc


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

    def __init__(self, opts, host="127.0.0.1", port=2000, tm_port=8000, euler2vec='none'):
        self.opts = opts

        # default seed
        self.random_generator = random.Random(None)
        self.np_random_generator = np.random.default_rng(None)
        # Connect to the CARLA simulator
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.opts["map"])

        # spectator
        # spectator = self.world.get_spectator()
        # spectator.set_transform(carla.Transform(carla.Location(*self.opts['cam_pos_lst'][0]),
        #                         carla.Rotation(*self.opts['cam_dir_lst'][0])))

        # CarlaX is xy indexing; x,y (w,h) (n_col,n_row)
        # x_min, x_max, _, _, _, _ = opts["spawn_area"]
        # self.map_width = x_max - x_min
        # print(self.map_width)

        self.num_cam = self.opts["num_cam"]
        # have seen how many cameras: step = {0,...,num_cam}
        # step = {0,...,num_cam-1}: input state to policy network
        # step = num_cam: done = True
        self.step_counter = 0

        self.action_names = self.opts['env_action_space']  # might be in [x,y,z,dir_x,dir_y,dir_z,roll,fov]
        self.decoded_action_names = self.opts['env_action_space'].split('-')  # [x,y,z,yaw,pitch,roll,fov] = cfg
        config_dict = {'x': 0, 'y': 1, 'z': 2, 'yaw': 3, 'pitch': 4, 'roll': 5, 'fov': 6}
        action_ids = [config_dict[key] for key in self.decoded_action_names]
        self.use_default = (torch.arange(len(config_dict))[:, None] == torch.tensor(action_ids)[None]).sum(dim=1).bool()
        self.euler2vec = [angle for angle in euler2vec.split('-') if angle in self.decoded_action_names]
        if 'yaw' in self.euler2vec:
            self.action_names = self.action_names.replace('yaw', 'dir_x-dir_y')
            if 'pitch' in self.euler2vec:
                self.action_names = self.action_names.replace('pitch', 'dir_z')
                config_dict = {'x': 0, 'y': 1, 'z': 2, 'dir_x': 3, 'dir_y': 4, 'dir_z': 5, 'roll': 6, 'fov': 7}
            else:
                config_dict = {'x': 0, 'y': 1, 'z': 2, 'dir_x': 3, 'dir_y': 4, 'pitch': 5, 'roll': 6, 'fov': 7}
        else:
            assert len(self.euler2vec) == 0  # if converted, then must include yaw
        self.config_dim = len(config_dict)
        self.action_names = self.action_names.split('-')
        action_ids = [config_dict[key] for key in self.action_names]
        self.action2config = torch.arange(len(config_dict))[:, None] == torch.tensor(action_ids)[None]
        x_min, x_max, y_min, y_max, z_min, z_max, pitch_min, pitch_max, fov_min, fov_max = opts["camera_range"]
        self.cfg_weight = torch.tensor([(x_max - x_min) / 2, (y_max - y_min) / 2, (z_max - z_min) / 2,
                                        180, (pitch_max - pitch_min) / 2, 180, (fov_max - fov_min) / 2])
        self.cfg_bias = torch.tensor([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2,
                                      0, (pitch_max + pitch_min) / 2, 0, (fov_max + fov_min) / 2])

        # Define any other attributes or variables needed for your environment
        # turn on sync mode
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(100.0)

        # world actors
        self.camera_configs = {}
        self.camera_intrinsics = {}
        self.camera_extrinsics = {}
        self.cameras = {}
        self.img_cam_buffer = {}
        self.pedestrians = []
        self.pedestrian_gts = []

        # avoid getting blueprints too often
        # https://github.com/carla-simulator/carla/issues/3197#issuecomment-1113692585
        self.camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.camera_bp.set_attribute("image_size_x", str(self.opts["cam_x"]))
        self.camera_bp.set_attribute("image_size_y", str(self.opts["cam_y"]))
        self.camera_bp.set_attribute("fov", str(self.opts["cam_fov"]))
        self.pedestrian_bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        self.walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

    # action is for a SINGLE camera
    # convert cfg: location/rotation/fov from [-1, 1] to different ranges
    # [x, y, z, yaw, pitch, roll, fov] = cfg
    # x, y, z \in [x_min, x_max], [y_min, y_max], [z_min, z_max]
    # pitch, yaw, roll \in [-90, 90], [-180, 180], [-180, 180]
    # fov \in [0, 180]
    # dir_x = torch.cos(yaw) * torch.cos(pitch)
    # dir_y = torch.sin(yaw) * torch.cos(pitch)
    # dir_z = torch.sin(pitch)
    # vec = vec.norm(dim=-1)
    # pitch = torch.arcsin(vec[..., 2])
    # yaw = torch.atan2(vec[..., 1], vec[..., 0])
    def encode_camera_cfg(self, cfg):
        is_tensor = isinstance(cfg, torch.Tensor)
        device = cfg.device if is_tensor else 'cpu'
        cfg = torch.tensor(cfg, dtype=torch.float) if not is_tensor else cfg  # \in opts["camera_range"]
        _cfg = (cfg - self.cfg_bias.to(device)) / self.cfg_weight.to(device)  # \in [-1,1]
        if 'yaw' in self.euler2vec:
            [x, y, z, yaw, pitch, roll, fov] = _cfg
            _yaw = torch.deg2rad(yaw * self.cfg_weight[3].to(device) + self.cfg_bias[3].to(device))
            _pitch = torch.deg2rad(pitch * self.cfg_weight[4].to(device) + self.cfg_bias[4].to(device))
            if 'pitch' in self.euler2vec:
                dir_x = torch.cos(_yaw) * torch.cos(_pitch)
                dir_y = torch.sin(_yaw) * torch.cos(_pitch)
                dir_z = torch.sin(_pitch)
                action = [x, y, z, dir_x, dir_y, dir_z, roll, fov]
            else:
                dir_x = torch.cos(_yaw)
                dir_y = torch.sin(_yaw)
                action = [x, y, z, dir_x, dir_y, pitch, roll, fov]
            action = torch.stack(action, dim=-1)  # one more dimension
        else:
            action = _cfg
        return action if is_tensor else action.tolist()

    def decode_camera_cfg(self, action):
        device = action.device if isinstance(action, torch.Tensor) else 'cpu'
        if 'yaw' in self.euler2vec:
            if 'pitch' in self.euler2vec:
                [x, y, z, dir_x, dir_y, dir_z, roll, fov] = action
                vec = F.normalize(torch.stack([dir_x, dir_y, dir_z], dim=-1), dim=-1)
                pitch = torch.rad2deg(torch.arcsin(vec[..., 2]))
                yaw = torch.rad2deg(torch.atan2(vec[..., 1], vec[..., 0]))
                pitch = (pitch - self.cfg_bias[4].to(device)) / self.cfg_weight[4].to(device)
            else:
                [x, y, z, dir_x, dir_y, pitch, roll, fov] = action
                vec = F.normalize(torch.stack([dir_x, dir_y], dim=-1), dim=-1)
                yaw = torch.rad2deg(torch.atan2(vec[..., 1], vec[..., 0]))
            yaw = (yaw - self.cfg_bias[3].to(device)) / self.cfg_weight[3].to(device)
            _cfg = torch.tensor([x, y, z, yaw, pitch, roll, fov], device=device)  # one more dimension
        else:
            _cfg = action
        return _cfg * self.cfg_weight.to(device) + self.cfg_bias.to(device)

    def action(self, act, cam):
        device = act.device if isinstance(act, torch.Tensor) else 'cpu'
        # camera config for the next camera
        # allow more flexible choice of action space (x-y-z-pitch-yaw-roll-fov)
        # convert normalised action space to unnormalised ones
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act, dtype=torch.float, device=device)
        _action = self.action2config.to(device).float() @ act
        _action = torch.clamp(_action, -1, 1)
        cfg = self.decode_camera_cfg(_action)
        # default settings for limited action space
        location, rotation, fov = torch.tensor(self.opts["cam_pos_lst"], device=device)[cam], \
            torch.tensor(self.opts["cam_dir_lst"], device=device)[cam], \
            torch.tensor(self.opts["cam_fov"], device=device).reshape([-1])
        default_cfg = torch.cat([location, rotation, fov])
        cfg = cfg * self.use_default.to(device) + default_cfg * ~self.use_default.to(device)
        return cfg

    def reset(self, seed=None, motion=False):
        # if a new seed is provided, set generator to used new seed
        # otherwise use old seed
        self.random_generator = random.Random(seed)
        self.np_random_generator = np.random.default_rng(seed)

        # Reset the environment to its initial state and return the initial observation
        self.respawn_pedestrians(n_chatgroup=self.opts['n_chatgroup'], n_walk=self.opts['n_walk'],
                                 motion=self.opts['motion'])
        self.reset_cameras()
        self.step_counter = 0

        # time.sleep(SLEEP_TIME)

        # NOTE: render all cameras by default
        observation = {
            "images": self.render(),
            "camera_configs": {cam: self.encode_camera_cfg(self.camera_configs[cam])
                               for cam in range(self.num_cam)},
            "step": self.step_counter
        }
        self.update_pedestrian_gts()
        info = {"pedestrian_gts": self.pedestrian_gts,
                "camera_intrinsics": self.camera_intrinsics,
                "camera_extrinsics": self.camera_extrinsics}  # Set any additional information

        # NOTE: Remember that Python only returns a reference to these objects
        # you may need to use copy.deepcopy() to avoid effects from further steps
        return observation, info

    def step(self, action):
        # Perform one step in the environment based on the given action
        # have seen self.step_counter cameras
        cam = self.step_counter
        cfg = self.action(action, cam).numpy().tolist()
        # the input action would be an array of 7 numbers, as defined in action space
        # values are in the range of 0-1
        loc = carla.Location(*cfg[:3])
        # cam_dir=[yaw, pitch, roll], carla.Rotation([pitch, yaw, roll])
        rot = carla.Rotation(cfg[4], cfg[3], cfg[5])
        fov = cfg[6]
        new_transform = carla.Transform(loc, rot)
        if float(self.cameras[cam].attributes["fov"]) != fov:
            # change camera fov, first destroy the old camera
            self.cameras[cam].destroy()
            # create new camera blueprint
            self.camera_bp.set_attribute("fov", str(fov))
            # spawn the camera
            camera = self.world.spawn_actor(self.camera_bp, new_transform)
            # record camera related information
            self.cameras[cam] = camera
        else:
            # update the camera transform
            self.cameras[cam].set_transform(new_transform)

        # update camera mats
        cam_config, intrinsic, extrinsic = get_camera_config(self.opts["cam_x"], self.opts["cam_y"], loc, rot, fov)
        self.camera_configs[cam] = cam_config
        self.camera_intrinsics[cam] = intrinsic
        self.camera_extrinsics[cam] = extrinsic

        time.sleep(SLEEP_TIME)

        self.step_counter += 1
        # Update the state, calculate the reward, and check for termination
        # Set the current observation
        observation = {
            "images": self.render(),
            "camera_configs": {cam: self.encode_camera_cfg(self.camera_configs[cam])
                               for cam in range(self.num_cam)},
            "step": self.step_counter
        }
        self.update_pedestrian_gts()

        # Set the reward for the current step
        reward = 0
        # Set condition for the end of episode: after a fixed number of step() call
        done = self.step_counter >= self.num_cam  # Set whether the episode has terminated or not
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
        for pedestrian in self.pedestrians:
            if 'controller' in pedestrian:
                ai_controller = self.world.get_actor(pedestrian['controller'])
                ai_controller.stop()
                destroyed_successfully = ai_controller.destroy()
            actor = self.world.get_actor(pedestrian['id'])
            destroyed_successfully = actor.destroy()
        for camera in self.cameras.values():
            camera.destroy()

    def respawn_pedestrians(self, n_chatgroup=4, chatgroup_size=(2, 4), chatgroup_radius=(0.5, 1.5),
                            n_walk=15, n_roam=0, percentagePedestriansRunning=0.2, motion=False):
        # Destroy existing actors, create new ones randomly
        for pedestrian in self.pedestrians:
            if 'controller' in pedestrian:
                ai_controller = self.world.get_actor(pedestrian['controller'])
                ai_controller.stop()
                destroyed_successfully = ai_controller.destroy()
            actor = self.world.get_actor(pedestrian['id'])
            destroyed_successfully = actor.destroy()
        self.pedestrians = []
        # spawn parameter, make the spawn area 0.5m smaller
        min_x, max_x = self.opts["spawn_area"][0:2]
        min_y, max_y = self.opts["spawn_area"][2:4]
        min_x, min_y = min_x + 0.5, min_y + 0.5
        max_x, max_y = max_x - 0.5, max_y - 0.5
        # 1. take all the random locations to spawn
        spawn_points = {'chat': [], 'walk': [], 'roam': []}
        # chat
        for _ in range(n_chatgroup):
            group_center_x = self.random_generator.uniform(min_x, max_x)
            group_center_y = self.random_generator.uniform(min_y, max_y)
            group_size = self.random_generator.randint(chatgroup_size[0], chatgroup_size[1])
            group_radius = self.random_generator.uniform(chatgroup_radius[0], chatgroup_radius[1])
            for _ in range(group_size):
                offset_x = self.random_generator.uniform(-group_radius, group_radius)
                offset_y = self.random_generator.uniform(-group_radius, group_radius)

                spawn_x = min(max(group_center_x + offset_x, min_x), max_x)
                spawn_y = min(max(group_center_y + offset_y, min_y), max_y)
                loc = carla.Location(spawn_x, spawn_y, 1.0)
                rot = carla.Rotation(0, math.degrees(math.atan2(-offset_y, -offset_x)), 0)
                spawn_point = carla.Transform(loc, rot)
                spawn_points['chat'].append(spawn_point)
        # walk
        for _ in range(n_walk):
            spawn_x = self.random_generator.uniform(min_x, max_x)
            spawn_y = self.random_generator.uniform(min_y, max_y)
            loc = carla.Location(spawn_x, spawn_y, 1.0)
            rot = carla.Rotation(0, self.random_generator.random() * 360, 0)
            spawn_point = carla.Transform(loc, rot)
            spawn_points['walk'].append(spawn_point)
        # roam
        for _ in range(n_roam):
            spawn_x = self.random_generator.uniform(min_x, max_x)
            spawn_y = self.random_generator.uniform(min_y, max_y)
            loc = carla.Location(spawn_x, spawn_y, 1.0)
            rot = carla.Rotation(0, self.random_generator.random() * 360, 0)
            spawn_point = carla.Transform(loc, rot)
            spawn_points['roam'].append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        types = []
        for pattern in spawn_points.keys():
            for spawn_point in spawn_points[pattern]:
                walker_bp = self.random_generator.choice(self.pedestrian_bps)
                # make sure all pedestrians are vincible
                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")
                # set the max speed
                if pattern == 'chat' or not walker_bp.has_attribute('speed'):
                    walker_speed.append(0.0)
                else:
                    if (random.random() > percentagePedestriansRunning):
                        # walking
                        walker_speed.append(float(walker_bp.get_attribute('speed').recommended_values[1]))
                    else:
                        # running
                        walker_speed.append(float(walker_bp.get_attribute('speed').recommended_values[2]))
                batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
                types.append(pattern)
        # apply spawn pedestrian
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if not results[i].error:
                # if error happens, very likely to be spawning failure caused by collision
                self.pedestrians.append({'id': results[i].actor_id,
                                         'type': types[i],
                                         'speed': walker_speed[i]})
        # print(f"{len(self.pedestrians)} pedestrians spawned")
        if motion:
            # 3. we spawn the walker controller
            batch = []
            for i in range(len(self.pedestrians)):
                batch.append(carla.command.SpawnActor(self.walker_controller_bp, carla.Transform(),
                                                      self.pedestrians[i]['id']))
            results = self.client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if not results[i].error:
                    self.pedestrians[i]["controller"] = results[i].actor_id
            # 4. we put together the walkers and controllers id to get the objects from their id
            # wait for a tick to ensure client receives the last transform of the walkers we have just created
            self.world.tick()

            # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
            for pedestrian in self.pedestrians:
                ai_controller = self.world.get_actor(pedestrian['controller'])
                ai_controller.start()
                # start walking
                if pedestrian['type'] != 'chat' and motion:
                    # set walk to random point
                    destination_x = self.random_generator.uniform(min_x, max_x)
                    destination_y = self.random_generator.uniform(min_y, max_y)
                    destination = carla.Location(destination_x, destination_y, 0.22)
                    # all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
                    ai_controller.go_to_location(destination)
                    # max speed
                    ai_controller.set_max_speed(pedestrian['speed'])
        pass

    def update_pedestrian_gts(self):
        self.pedestrian_gts = []
        for pedestrian in self.pedestrians:
            # 1. new ground truth format
            actor = self.world.get_actor(pedestrian['id'])
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
                    "id": pedestrian['id'],
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
            p = self.camera_intrinsics[cam] @ self.camera_extrinsics[cam] @ vert_homo
            # Check if in front of the camera
            if p[-1] > 0:
                p = p[:2] / p[-1]
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

        if cfg is None:
            locations = np.array(self.opts["cam_pos_lst"])
            rotations = np.array(self.opts["cam_dir_lst"])
            fovs = float(self.opts["cam_fov"]) * np.ones(self.num_cam)
        else:
            locations, rotations, fovs = cfg[:, :3], cfg[:, 3:6], cfg[:, 6]

        for cam, (cam_pos, cam_dir, fov) in enumerate(zip(locations, rotations, fovs)):
            if float(fov) != float(self.camera_bp.get_attribute("fov")):
                self.camera_bp.set_attribute("fov", str(fov))
            loc = carla.Location(*cam_pos)
            # cam_dir=[yaw, pitch, roll], carla.Rotation([pitch, yaw, roll])
            rot = carla.Rotation(*cam_dir[[1, 0, 2]])
            camera_init_trans = carla.Transform(loc, rot)
            # spawn the camera
            self.cameras[cam] = self.world.spawn_actor(self.camera_bp, camera_init_trans)
            self.img_cam_buffer[cam] = Queue(maxsize=0)

            # record camera related information
            # save camera configs, rather than projection matrices
            # projection/intrinsic/extrinsic matrices can be built from configs
            cam_config, intrinsic, extrinsic = get_camera_config(self.opts["cam_x"], self.opts["cam_y"], loc, rot, fov)
            self.camera_configs[cam] = cam_config
            self.camera_intrinsics[cam] = intrinsic
            self.camera_extrinsics[cam] = extrinsic


def process_img(img):
    # img is carla.libcarla.Image object;
    # returns a ndarray
    img_bgra = np.reshape(np.copy(img.raw_data), (img.height, img.width, 4))
    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
    return img_rgb


def get_camera_config(image_w, image_h, loc, rot, fov):
    # image_w = int(camera.attributes["image_size_x"])
    # image_h = int(camera.attributes["image_size_y"])
    # fov = float(camera.attributes["fov"])
    #
    # transform = camera.get_transform()
    # loc = transform.location
    # rot = transform.rotation

    f = image_w / (2.0 * np.tan(fov * np.pi / 360))
    Cx = image_w / 2.0
    Cy = image_h / 2.0

    # camera value is consist of 9 elements
    cam_value = {
        "x": loc.x,
        "y": loc.y,
        "z": loc.z,
        "pitch": rot.pitch,
        "yaw": rot.yaw,
        "roll": rot.roll,
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
    import torchvision.transforms as T

    container = docker_run_carla(1)

    with open('cfg/RL/town05market.cfg', "r") as fp:
        dataset_config = json.load(fp)
    # dataset_config['motion'] = True
    # dataset_config['n_chatgroup'] = 4
    transform = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

    env = CarlaCameraSeqEnv(dataset_config, port=2000, tm_port=8000, euler2vec='yaw-pitch')
    done = False
    for i in tqdm(range(4)):
        _observation, info = env.reset(motion=True)
        print(_observation['step'])
        j = 0
        while not done:
            action = env.action2config.float().T @ torch.tensor(_observation['camera_configs'][j])
            observation, reward, done, info = env.step(action)
            print(observation['step'])
            j += 1
        done = False

    # in loop listen
    # env.reset()
    # t0 = time.time()
    # for i in tqdm(range(400)):
    #     for j in range(NUM_TICKS):
    #         env.world.tick()
    #     env.render()
    # print(f'in-loop listen time: {time.time() - t0}')

    container.stop()
