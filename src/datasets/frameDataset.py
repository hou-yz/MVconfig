import os
import re
import json
import time
from operator import itemgetter
from PIL import Image
from kornia.geometry import warp_perspective
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
from src.utils.projection import *
from src.utils.image_utils import draw_umich_gaussian, random_affine
import matplotlib.pyplot as plt


def get_centernet_gt(Rshape, x_s, y_s, v_s, w_s=None, h_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    reg_mask = np.zeros([top_k], dtype=bool)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret


def read_pom(root):
    bbox_by_pos_cam = {}
    cam_pos_pattern = re.compile(r'(\d+) (\d+)')
    cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
    with open(os.path.join(root, 'rectangles.pom'), 'r') as fp:
        for line in fp:
            if 'RECTANGLE' in line:
                cam, pos = map(int, cam_pos_pattern.search(line).groups())
                if pos not in bbox_by_pos_cam:
                    bbox_by_pos_cam[pos] = {}
                if 'notvisible' in line:
                    bbox_by_pos_cam[pos][cam] = None
                else:
                    cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                    bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                 min(right, 1920 - 1), min(bottom, 1080 - 1)]
    return bbox_by_pos_cam


class frameDataset(VisionDataset):
    def __init__(self, base, split='train', reID=False, world_reduce=4, img_reduce=12,
                 world_kernel_size=10, img_kernel_size=10,
                 split_ratio=(0.8, 0.1, 0.1), top_k=100, force_download=True, augmentation=False,
                 interactive=False):
        super().__init__(base.root)

        self.base = base
        self.num_cam, self.num_frame = base.num_cam, base.num_frame
        # world (grid) reduce: on top of the 2.5cm grid
        self.reID, self.top_k = reID, top_k
        # reduce = input/output
        self.world_reduce, self.img_reduce = world_reduce, img_reduce
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    T.Resize((np.array(self.img_shape) * 8 // self.img_reduce).tolist())])
        self.augmentation = augmentation
        self.interactive = interactive

        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()

        # split = ('train', 'val', 'test'), split_ratio=(0.8, 0.1, 0.1)
        split_ratio = tuple(sum(split_ratio[:i + 1]) for i in range(len(split_ratio)))
        # assert split_ratio[-1] == 1
        self.split = split
        if split == 'train':
            frame_range = range(0, int(self.num_frame * split_ratio[0]))
        elif split == 'val':
            frame_range = range(int(self.num_frame * split_ratio[0]), int(self.num_frame * split_ratio[1]))
        elif split == 'trainval':
            frame_range = range(0, int(self.num_frame * split_ratio[1]))
        elif split == 'test':
            frame_range = range(int(self.num_frame * split_ratio[1]), self.num_frame)
        else:
            raise Exception

        # gt in mot format for evaluation
        self.gt_fname = f'{self.root}/gt.txt'
        if self.base.__name__ == 'CarlaX':
            # generate same pedestrian layout for the same frame
            self.fixed_seeds = np.random.randint(409600, size=self.num_frame)[frame_range]
            self.frames = list(frame_range)
            self.world_gt = {}
            self.gt_array = np.array([]).reshape([0, 3])
            self.config_dim = base.env.observation_space['camera_configs'].shape[0]
            self.action_dim = base.env.action_space.shape[0] if interactive else None
        else:
            # get camera matrices
            self.proj_mats = self.get_world_imgs_trans()
            # get image & world coverage masks
            # world_masks = torch.ones([self.num_cam, 1] + self.worldgrid_shape)
            # self.imgs_region = warp_perspective(world_masks, torch.inverse(self.proj_mats), self.img_shape, 'nearest')
            # img_masks = torch.ones([self.num_cam, 1, self.base.img_shape[0], self.base.img_shape[1]])
            # self.Rworld_coverage = warp_perspective(img_masks, self.proj_mats, self.Rworld_shape)

            self.img_fpaths = self.get_image_fpaths(frame_range)
            self.world_gt, self.imgs_gt, self.pid_dict, self.frames = self.get_gt_targets(
                split if split == 'trainval' else f'{split} \t', frame_range)
            if not os.path.exists(self.gt_fname) or force_download:
                og_gt = [[] for _ in range(self.num_cam)]
                for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
                    frame = int(fname.split('.')[0])
                    with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                        all_pedestrians = json.load(json_file)
                    for pedestrian in all_pedestrians:
                        def is_in_cam(cam, grid_x, grid_y):
                            visible = not (pedestrian['views'][cam]['xmin'] == -1 and
                                           pedestrian['views'][cam]['xmax'] == -1 and
                                           pedestrian['views'][cam]['ymin'] == -1 and
                                           pedestrian['views'][cam]['ymax'] == -1)
                            in_view = (pedestrian['views'][cam]['xmin'] > 0 and
                                       pedestrian['views'][cam]['xmax'] < 1920 and
                                       pedestrian['views'][cam]['ymin'] > 0 and
                                       pedestrian['views'][cam]['ymax'] < 1080)

                            # Rgrid_x, Rgrid_y = grid_x // self.world_reduce, grid_y // self.world_reduce
                            # in_map = Rgrid_x < self.Rworld_shape[0] and Rgrid_y < self.Rworld_shape[1]
                            return visible and in_view

                        grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                        for cam in range(self.num_cam):
                            if is_in_cam(cam, grid_x, grid_y):
                                og_gt[cam].append(np.array([frame, grid_x, grid_y]))
                og_gt = [np.stack(og_gt[cam], axis=0) for cam in range(self.num_cam)]
                np.savetxt(self.gt_fname, np.unique(np.concatenate(og_gt, axis=0), axis=0), '%d')
                for cam in range(self.num_cam):
                    np.savetxt(f'{self.gt_fname}.{cam}', og_gt[cam], '%d')
            self.gt_array = np.loadtxt(self.gt_fname)
            self.config_dim = None
            self.action_dim = None
        pass

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_gt_targets(self, split, frame_range):
        num_world_bbox, num_imgs_bbox = 0, 0
        world_gt = {}
        imgs_gt = {}
        pid_dict = {}
        frames = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                frames.append(frame)
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                for pedestrian in all_pedestrians:
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    if pedestrian['personID'] not in pid_dict:
                        pid_dict[pedestrian['personID']] = len(pid_dict)
                    num_world_bbox += 1
                    if self.base.indexing == 'xy':
                        world_pts.append((grid_x, grid_y))
                    else:
                        world_pts.append((grid_y, grid_x))
                    world_pids.append(pid_dict[pedestrian['personID']])
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pid_dict[pedestrian['personID']])
                            num_imgs_bbox += 1
                world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))

        print(f'{split}:\t pid: {len(pid_dict)}, frame: {len(frames)}, '
              f'world bbox: {num_world_bbox / len(frames):.1f}, '
              f'imgs bbox per cam: {num_imgs_bbox / len(frames) / self.num_cam:.1f}')
        return world_gt, imgs_gt, pid_dict, frames

    def get_carla_gt_targets(self, all_pedestrians):
        world_pts, world_lwh, world_pids = [], [], []
        img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
        for pedestrian in all_pedestrians:
            # world_pts.append([pedestrian["x"], pedestrian["y"], pedestrian["z"]])
            world_pts.append(self.base.get_worldgrid_from_worldcoord(
                np.array([pedestrian["x"], pedestrian["y"]])[:, None])[:, 0])
            world_lwh.append([pedestrian["l"], pedestrian["w"], pedestrian["h"]])
            world_pids.append(pedestrian["id"])
            for cam in range(self.num_cam):
                if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                    img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]))
                    img_pids[cam].append(pedestrian["id"])
        for cam in range(self.num_cam):
            img_bboxs[cam], img_pids[cam] = np.array(img_bboxs[cam]), np.array(img_pids[cam])
        return np.array(world_pts), np.array(world_lwh), np.array(world_pids), img_bboxs, img_pids

    def get_world_imgs_trans(self, z=0):
        # image and world feature maps from xy indexing, change them into world (xy/ij) indexing / image (xy) indexing
        world_zoom_mat = np.diag([1 / self.world_reduce, 1 / self.world_reduce, 1])
        Rworldgrid_from_worldcoord = world_zoom_mat @ \
                                     self.base.world_indexing_from_xy_mat @ \
                                     np.linalg.inv(self.base.worldcoord_from_worldgrid_mat)

        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord = [get_worldcoord_from_imgcoord_mat(self.base.intrinsic_matrices[cam],
                                                                     self.base.extrinsic_matrices[cam],
                                                                     z / self.base.worldcoord_unit)
                                    for cam in range(self.num_cam)]
        # worldgrid(xy)_from_img(xy)
        Rworldgrid_from_imgcoord = torch.stack([torch.from_numpy(Rworldgrid_from_worldcoord @
                                                                 worldcoord_from_imgcoord[cam]).float()
                                                for cam in range(self.num_cam)])
        return Rworldgrid_from_imgcoord

    def __getitem__(self, index, visualize=False):
        self.cur_frame = frame = self.frames[index]
        if self.base.__name__ == 'CarlaX':
            observation, info = self.base.env.reset(seed=self.fixed_seeds[index])
            self.base.intrinsic_matrices, self.base.extrinsic_matrices = \
                self.base.env.camera_intrinsics, self.base.env.camera_extrinsics
            # get camera matrices
            self.proj_mats = self.get_world_imgs_trans()
            # get image & world coverage masks
            # world_masks = torch.ones([self.num_cam, 1] + self.worldgrid_shape)
            # self.imgs_region = warp_perspective(world_masks, torch.inverse(self.proj_mats), self.img_shape, 'nearest')
            # img_masks = torch.ones([self.num_cam, 1, self.base.img_shape[0], self.base.img_shape[1]])
            # self.Rworld_coverage = warp_perspective(img_masks, self.proj_mats, self.Rworld_shape)

            imgs = observation["images"]
            configs = observation["camera_configs"]
            step_counter = observation["step"]
            world_pts, world_lwh, world_pids, img_bboxs, img_pids = self.get_carla_gt_targets(info["pedestrian_gts"])
            world_pts, world_pids = world_pts[:, :2], world_pids

            # record world gt
            if frame not in self.world_gt:
                self.world_gt[frame] = (world_pts, world_pids)
                self.gt_array = np.concatenate([self.gt_array,
                                                np.concatenate([frame * np.ones([len(world_pts), 1]),
                                                                world_pts], axis=1)],
                                               axis=0)
                # np.savetxt(self.gt_fname, np.concatenate(self.gt_array, axis=0), '%d')
            # else:
            #     assert (self.world_gt[frame][0] == world_pts).all()
        else:
            imgs = {cam: np.array(Image.open(self.img_fpaths[cam][frame]).convert('RGB'))
                    for cam in range(self.num_cam)}
            configs = None
            step_counter = None
            img_bboxs, img_pids = zip(*self.imgs_gt[frame].values())
            world_pts, world_pids = self.world_gt[frame]
        return self.prepare_gt(imgs, step_counter, configs, world_pts, world_pids, img_bboxs, img_pids, visualize)

    def step(self, action, visualize=False):
        observation, reward, done, info = self.base.env.step(action)

        # get camera matrices
        self.base.intrinsic_matrices, self.base.extrinsic_matrices = \
            self.base.env.camera_intrinsics, self.base.env.camera_extrinsics
        self.proj_mats = self.get_world_imgs_trans()
        # get image & world coverage masks
        world_masks = torch.ones([self.num_cam, 1] + self.worldgrid_shape)
        self.imgs_region = warp_perspective(world_masks, torch.inverse(self.proj_mats), self.img_shape, 'nearest')
        img_masks = torch.ones([self.num_cam, 1, self.base.img_shape[0], self.base.img_shape[1]])
        self.Rworld_coverage = warp_perspective(img_masks, self.proj_mats, self.Rworld_shape)

        imgs = observation["images"]
        configs = observation["camera_configs"]
        step_counter = observation["step"]
        world_pts, world_lwh, world_pids, img_bboxs, img_pids = self.get_carla_gt_targets(info["pedestrian_gts"])
        world_pts, world_pids = world_pts[:, :2], world_pids
        return self.prepare_gt(imgs, step_counter, configs, world_pts, world_pids, img_bboxs, img_pids, visualize), done

    def prepare_gt(self, imgs, step_counter, configs, world_pts, world_pids, img_bboxs, img_pids, visualize=False):
        def plt_visualize():
            import cv2
            from matplotlib.patches import Circle
            # fig, ax = plt.subplots(1)
            # ax.imshow(img)
            # for i in range(len(img_x_s)):
            #     x, y = img_x_s[i], img_y_s[i]
            #     if x > 0 and y > 0:
            #         ax.add_patch(Circle((x, y), 10))
            # plt.show()
            img0 = img.copy()
            for bbox in cam_img_bboxs:
                bbox = tuple(int(pt) for pt in bbox)
                cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            plt.imshow(img0)
            plt.show()

        if self.interactive and step_counter is not None:
            imgs = {step_counter: imgs[step_counter]}
            configs = np.array([config for cam, config in configs.items()], dtype=np.float32)
            # IMPORTANT: mask out padded items
            padding_mask = np.arange(self.num_cam) > step_counter
            configs[padding_mask] = -3
        else:
            step_counter = self.num_cam
            configs = np.zeros([self.num_cam, 0])

        aug_imgs, aug_imgs_gt, aug_mats, aug_masks = [], [], [], []
        for cam, img in imgs.items():
            cam_img_bboxs, cam_img_pids = img_bboxs[cam], img_pids[cam]
            if len(cam_img_bboxs.shape) == 1:
                cam_img_bboxs = cam_img_bboxs.reshape([-1, 4])
            if self.augmentation:
                img, cam_img_bboxs, cam_img_pids, M = random_affine(img, cam_img_bboxs, cam_img_pids)
            else:
                M = np.eye(3)
            aug_imgs.append(self.transform(img))
            aug_mats.append(torch.from_numpy(M).float())
            img_x_s, img_y_s = (cam_img_bboxs[:, 0] + cam_img_bboxs[:, 2]) / 2, cam_img_bboxs[:, 3]
            img_w_s, img_h_s = (cam_img_bboxs[:, 2] - cam_img_bboxs[:, 0]), (cam_img_bboxs[:, 3] - cam_img_bboxs[:, 1])

            img_gt = get_centernet_gt(self.Rimg_shape, img_x_s, img_y_s, cam_img_pids, img_w_s, img_h_s,
                                      reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            aug_imgs_gt.append(img_gt)
            # TODO: check the difference between different dataloader iteration
            if visualize:
                plt_visualize()

        aug_imgs = torch.stack(aug_imgs)
        aug_mats = torch.stack(aug_mats)
        aug_imgs_gt = {key: torch.stack([img_gt[key] for img_gt in aug_imgs_gt]) for key in aug_imgs_gt[0]}
        # world gt
        world_gt = get_centernet_gt(self.Rworld_shape, world_pts[:, 0], world_pts[:, 1], world_pids,
                                    reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)
        return (step_counter, configs, aug_imgs, aug_mats, self.proj_mats[list(imgs.keys())],
                world_gt, aug_imgs_gt, self.cur_frame)

    def __len__(self):
        return len(self.frames)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.datasets.wildtrack import Wildtrack
    from src.datasets.multiviewx import MultiviewX
    from src.datasets.carlax import CarlaX
    import random

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), force_download=True)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), force_download=True)
    import json

    with open('./cfg/RL/1.cfg', "r") as fp:
        dataset_config = json.load(fp)
    # dataset = frameDataset(CarlaX(dataset_config), split_ratio=(0.01, 0.1, 0.1))
    dataset = frameDataset(CarlaX(dataset_config, seed=seed), split_ratio=(0.01, 0.1, 0.1), interactive=True)
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), split='train')
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), split='train', semi_supervised=.1)
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), split='train', semi_supervised=0.5)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), split='train', semi_supervised=0.5)
    # min_dist = np.inf
    # for world_gt in dataset.world_gt.values():
    #     x, y = world_gt[0][:, 0], world_gt[0][:, 1]
    #     if x.size and y.size:
    #         xy_dists = ((x - x[:, None]) ** 2 + (y - y[:, None]) ** 2) ** 0.5
    #         np.fill_diagonal(xy_dists, np.inf)
    #         min_dist = min(min_dist, np.min(xy_dists))
    #         pass
    dataloader = DataLoader(dataset, 2, True, num_workers=0)
    t0 = time.time()
    # _ = next(iter(dataloader))
    for i in range(20):
        _ = dataset.__getitem__(i % len(dataset), visualize=True)
        if dataset.base.__name__ == 'CarlaX' and dataset.interactive:
            done = False
            while not done:
                _, done = dataset.step(np.random.randn(2), visualize=True)

    print(time.time() - t0)
    pass
    if False:
        import matplotlib.pyplot as plt
        from src.utils.projection import get_worldcoord_from_imagecoord

        world_grid_maps = []
        xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
        H, W = xx.shape
        image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
        for cam in range(dataset.num_cam):
            world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(),
                                                          dataset.base.intrinsic_matrices[cam],
                                                          dataset.base.extrinsic_matrices[cam])
            world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
            world_grid_map = np.zeros(dataset.worldgrid_shape)
            for i in range(H):
                for j in range(W):
                    x, y = world_grids[i, j]
                    if dataset.base.indexing == 'xy':
                        if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
                            world_grid_map[int(y), int(x)] += 1
                    else:
                        if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
                            world_grid_map[int(x), int(y)] += 1
            world_grid_map = world_grid_map != 0
            plt.imshow(world_grid_map)
            plt.show()
            world_grid_maps.append(world_grid_map)
            pass
        plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
        plt.show()
        pass
