import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset

intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']


class Wildtrack(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # image of shape C,H,W (C,N_row,N_col); xy indexing; x,y (w,h) (n_col,n_row)
        # WILDTRACK has xy-indexing: H*W=1440*480, thus x is \in [0,480), y \in [0,1440)
        # WILDTRACK has in-consistent unit: centi-meter (cm) for calibration & pos annotation
        self.__name__ = 'Wildtrack'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [1440, 480]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 7, 2000
        # unit in meters
        self.worldcoord_unit = 0.01
        self.worldcoord_from_worldgrid_mat = np.array([[2.5, 0, -300], [0, 2.5, -900], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    @staticmethod
    def get_worldgrid_from_pos(pos):
        grid_x = pos % 480
        grid_y = pos // 480
        return np.array([grid_x, grid_y])[None]

    @staticmethod
    def get_pos_from_worldgrid(grid):
        grid_x, grid_y = grid[:, 0], grid[:, 1]
        return grid_x + grid_y * 480

    @staticmethod
    def get_worldgrid_from_worldcoord(coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        coord_x, coord_y = coord[:, 0], coord[:, 1]
        grid_x = (coord_x + 300) / 2.5
        grid_y = (coord_y + 900) / 2.5
        return np.stack([grid_x, grid_y], axis=1).astype(int)

    @staticmethod
    def get_worldcoord_from_worldgrid(grid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = grid[:, 0], grid[:, 1]
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        return np.stack([coord_x, coord_y], axis=1)

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, coord):
        grid = self.get_worldgrid_from_worldcoord(coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=float).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix


def test():
    from src.utils.projection import get_worldcoord_from_imagecoord
    from src.datasets.frameDataset import read_pom
    dataset = Wildtrack(os.path.expanduser('~/Data/Wildtrack'), )
    pom = read_pom(dataset.root)

    for cam in range(dataset.num_cam):
        head_errors, foot_errors = [], []
        for pos in range(0, np.product(dataset.worldgrid_shape), 16):
            bbox = pom[pos][cam]
            foot_wc = dataset.get_worldcoord_from_pos(pos)
            if bbox is None:
                continue
            foot_ic = np.array([[(bbox[0] + bbox[2]) / 2, bbox[3]]])
            head_ic = np.array([[(bbox[0] + bbox[2]) / 2, bbox[1]]])
            p_foot_wc = get_worldcoord_from_imagecoord(foot_ic, dataset.intrinsic_matrices[cam],
                                                       dataset.extrinsic_matrices[cam])
            p_head_wc = get_worldcoord_from_imagecoord(head_ic, dataset.intrinsic_matrices[cam],
                                                       dataset.extrinsic_matrices[cam], z=1.8 / dataset.worldcoord_unit)
            head_errors.append(np.linalg.norm(p_head_wc - foot_wc))
            foot_errors.append(np.linalg.norm(p_foot_wc - foot_wc))
            pass

        print(f'average head error: {np.average(head_errors) * dataset.worldcoord_unit}, '
              f'average foot error: {np.average(foot_errors) * dataset.worldcoord_unit} (world meters)')
        pass
    pass


if __name__ == '__main__':
    test()
