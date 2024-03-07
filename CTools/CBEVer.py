import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from CTools.CVisualizer import CVisualizer
vis = CVisualizer()

class CBEVer():
    def __init__(self,xyz,img_size = [640,640],bound=[-25,25,-25,25,-2.73,1.27],sem=None,inst=None):
        self.bound = bound
        self.xyz = xyz
        self.sem = sem
        self.inst = inst
        self.img_size = img_size
        self.resolution = (self.bound[1] - self.bound[0]) / self.img_size[0]
    def extract_inlier(self):
        minX,maxX,minY,maxY,minZ,maxZ = self.bound
        mask = np.where(
            (self.xyz[:, 0] >= minX) &
            (self.xyz[:, 0] <= maxX) &
            (self.xyz[:, 1] >= minY) &
            (self.xyz[:, 1] <= maxY) &
            (self.xyz[:, 2] >= minZ) &
            (self.xyz[:, 2] <= maxZ)
        )
        self.xyz = self.xyz[mask]
        if self.sem is not None:
            self.sem = self.sem[mask]
        if self.inst is not None:
            self.inst = self.inst[mask]
        return mask

    def xyz_to_bev(self):
        Height = self.img_size[0] + 1
        Width = self.img_size[1] + 1
        xyz = np.copy(self.xyz)
        # xyz[:, 0] = np.int_(np.floor(xyz[:, 0] / self.resolution))
        # xyz[:, 1] = np.int_(np.floor(xyz[:, 1] / self.resolution) + Width / 2)
        xyz[:, 0] = np.int_(np.floor((xyz[:, 0] - np.min(xyz[:,0]))  / self.resolution))
        xyz[:, 1] = np.int_(np.floor((xyz[:, 1] - np.min(xyz[:,1])) / self.resolution))

        _, indices = np.unique(xyz[:, 0:2], axis=0, return_index=True)
        xyz_frac = xyz[indices]
        indices = np.lexsort((-xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        xyz = xyz[indices]
        height_map = np.zeros((self.img_size[0], self.img_size[1]))
        max_height = float(np.abs(self.bound[5] - self.bound[4]))
        height_map[np.int_(xyz_frac[:, 0]), np.int_(xyz_frac[:, 1])] = xyz_frac[:, 2] / max_height
        # height_map = plt.get_cmap('Blues')(height_map)[:, :, :3]
        height_map = (255 * height_map).astype(np.uint8)
        return height_map


    def xyz_sem_inst_to_bev(self):
        Height = self.img_size[0] + 1
        Width = self.img_size[1] + 1
        xyz = np.copy(self.xyz)
        sem = np.copy(self.sem)
        inst = np.copy(self.inst)
        # xyz[:, 0] = np.int_(np.floor(xyz[:, 0] / self.resolution))
        # xyz[:, 1] = np.int_(np.floor(xyz[:, 1] / self.resolution) + Width / 2)
        xyz[:, 0] = np.int_(np.floor((xyz[:, 0] - np.min(xyz[:,0]))  / self.resolution))
        xyz[:, 1] = np.int_(np.floor((xyz[:, 1] - np.min(xyz[:,1])) / self.resolution))

        _, indices = np.unique(xyz[:, 0:2], axis=0, return_index=True)
        xyz_frac = xyz[indices]
        sem_frac = sem[indices]
        inst_frac = inst[indices]
        indices = np.lexsort((-xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        xyz = xyz[indices]
        sem = sem[indices]
        inst = inst[indices]

        height_map = np.zeros((self.img_size[0], self.img_size[1]))
        max_height = float(np.abs(self.bound[5] - self.bound[4]))
        height_map[np.int_(xyz_frac[:, 0]), np.int_(xyz_frac[:, 1])] = xyz_frac[:, 2] / max_height
        # height_map = plt.get_cmap('Blues')(height_map)[:, :, :3]
        height_color_map = np.zeros((Height - 1, Width - 1, 3))
        height_color_map[:, :, 1] = height_map[0:self.img_size[0], 0:self.img_size[1]]
        height_color_map = (255 * height_color_map).astype(np.uint8)

        sem_color_map = np.zeros((Height - 1, Width - 1, 3))
        # sem_color = sem_color_lut[inv_lut[remap_lut[sem_frac]]]
        sem_color = vis.create_colors_from_labels(sem_frac)
        for i in range(3):  # r g b
            tmp_map = np.zeros((Height, Width))
            tmp_map[np.int_(xyz_frac[:, 0]), np.int_(xyz_frac[:, 1])] = sem_color[:, i]
            sem_color_map[:, :, i] = tmp_map[0:self.img_size[0], 0:self.img_size[1]]
        sem_color_map = (255 * sem_color_map).astype(np.uint8)


        inst_color_map = np.zeros((Height - 1, Width - 1, 3))
        inst_color = vis.create_colors_from_labels(inst_frac)
        for i in range(3):  # r g b
            tmp_map = np.zeros((Height, Width))
            tmp_map[np.int_(xyz_frac[:, 0]), np.int_(xyz_frac[:, 1])] = inst_color[:, i]
            inst_color_map[:, :, i] = tmp_map[0:self.img_size[0], 0:self.img_size[1]]
        inst_color_map = (255 * inst_color_map).astype(np.uint8)
        return height_color_map,sem_color_map,inst_color_map
    @staticmethod
    def combine_img_1x3(img1,img2,img3):
        combined_image = np.concatenate((img1, img2, img3), axis=1)
        return combined_image
