import copy

import open3d as o3d
import numpy as np
import yaml
import json
import os

usage = """
VisualizerKITTISemInst Usage
Keyboard Options:
  'N'         Next frame
  'B'         Previous frame
  'S'         Semantic mode
  'I'         Instance mode
  'M'         Motion mode
"""

class VisualizerKITTISemInst:
    def __init__(self, cfg_path):
        print(usage)
        self.pcd_extension = ['.bin']
        self.label_extension = ['.label']
        self.color_mode = "sem"
        self.color_mode = "inst"
        self.inst_color_dict = {}

        with open(cfg_path, 'r') as cfg_file:
            visual_cfg = yaml.safe_load(cfg_file)
        self.visual_cfg = visual_cfg
        self.bool_inv = self.visual_cfg['bool_inv']
        self.bool_map = self.visual_cfg['bool_map']
        self.target_label_list = self.visual_cfg['target_label_list']

        self.pcd_dir = [os.path.join(visual_cfg['pcd_dir'], item)
                        for item in os.listdir(visual_cfg['pcd_dir'])]
        self.pcd_extension = visual_cfg['pcd_extension']
        self.label_dir = [os.path.join(visual_cfg['label_dir'], item)
                          for item in os.listdir(visual_cfg['label_dir'])]
        self.pcd_dir.sort()
        self.label_dir.sort()
        self.label_extension = visual_cfg['label_extension']
        left = visual_cfg['position']['left']
        top = visual_cfg['position']['top']
        width = visual_cfg['position']['width']
        height = visual_cfg['position']['height']
        point_size = visual_cfg['point_size']

        dataset_cfg_path = visual_cfg['dataset_cfg_path']
        with open(dataset_cfg_path, 'r') as dataset_cfg_file:
            dataset_cfg = yaml.safe_load(dataset_cfg_file)
        self.dataset_cfg = dataset_cfg
        self.color_map = dataset_cfg['color_map']

        viewpoint_path = visual_cfg['viewpoint_cfg_path']
        with open(viewpoint_path, 'r') as viewpoint_file:
            viewpoint_cfg = json.load(viewpoint_file)
        front_vector = viewpoint_cfg['trajectory'][0]['front']
        lookat_vector = viewpoint_cfg['trajectory'][0]['lookat']
        up_vector = viewpoint_cfg['trajectory'][0]['up']
        zoom_vector = viewpoint_cfg['trajectory'][0]['zoom']

        self.remap_lut, self.inv_lut, self.sem_color_lut = self.__calc_lut()

        self.interval = visual_cfg['interval']
        self.index = visual_cfg['start_frame']
        if visual_cfg['end_frame'] == -1:
            self.end_frame = len(self.pcd_dir)
        else:
            self.end_frame = visual_cfg['end_frame']


        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name='semantic-kitti', width=width, height=height, left=left, top=top,
                               visible=True)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.vis.add_geometry(mesh_frame)

        self.__update_frame()
        self.vis.add_geometry(self.pcd)

        self.render_option = self.vis.get_render_option()
        self.render_option.background_color = np.asarray(self.visual_cfg['background_color'])
        self.render_option.point_size = point_size

        self.view_control = self.vis.get_view_control()
        self.view_control.set_front(front_vector)
        self.view_control.set_lookat(lookat_vector)
        self.view_control.set_up(up_vector)
        self.view_control.set_zoom(zoom_vector)

        self.vis.register_key_callback(ord('N'), self.__key_next_callback)
        self.vis.register_key_callback(ord('B'), self.__key_back_callback)
        self.vis.register_key_callback(ord('S'), self.__key_sem_callback)
        self.vis.register_key_callback(ord('I'), self.__key_inst_callback)
        self.vis.register_key_callback(ord('M'), self.__key_motion_callback)

    def __load_pc_bin(self, pcd_path):
        points = np.fromfile(pcd_path, dtype=np.float32)
        points = points.reshape((-1, 4))
        coordinate = points[:, 0:3]
        # remission = points[:, 3]
        self.pcd.points = o3d.utility.Vector3dVector(coordinate)

    def __calc_lut(self):
        DATA = self.dataset_cfg
        class_remap = DATA["learning_map"]
        class_inv_remap = DATA["learning_map_inv"]
        color_dict = DATA["color_map"]
        maxkey = max(class_remap.keys())
        remap_lut = np.zeros((maxkey + 10), dtype=np.int32)
        remap_lut[list(class_remap.keys())] = list(class_remap.values())

        maxkey = max(class_inv_remap.keys())
        inv_lut = np.zeros((maxkey + 10), dtype=np.int32)
        inv_lut[list(class_inv_remap.keys())] = list(class_inv_remap.values())
        max_sem_key = 0
        for key, data in color_dict.items():
            if key + 1 > max_sem_key:
                max_sem_key = key + 1
        sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)  # [max_sem_key+100,3]的nparray
        for key, value in color_dict.items():
            sem_color_lut[key] = np.array(value, np.float32) / 255.0  # 转换到[0,1]范围内

        return remap_lut, inv_lut, sem_color_lut

    def generate_color(self):
        return np.random.rand(3)

    def __create_colors_from_inst_labels_fixed(self,labels):
        unique_labels = np.unique(labels)
        colors = np.zeros((len(labels), 3))
        for i, instance_id in enumerate(labels):
            if instance_id not in self.inst_color_dict:
                self.inst_color_dict[instance_id] = self.generate_color()
            colors[i] = self.inst_color_dict[instance_id]

        for_mask = np.argwhere(labels == 0).squeeze(1)
        colors[for_mask] = np.array([1, 1, 1])
        return colors

    def __create_colors_from_inst_labels(self, labels):
        unique_labels = np.unique(labels)
        colors = np.zeros((len(labels), 3))
        for label in unique_labels:
            if label == -1:  # Ignore invalid label (if applicable)
                continue
            color = np.random.rand(3)  # Generate a random color for each instance
            mask = labels == label
            colors[mask] = color
        for_mask = np.argwhere(labels == 0).squeeze(1)
        colors[for_mask] = np.array([1, 1, 1])
        return colors


    def __load_label_bin(self, label_path):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        semantic_label = label & 0xFFFF
        target_label_list = self.target_label_list

        if self.bool_inv:
            semantic_label = self.inv_lut[semantic_label]
        if self.bool_map:
            semantic_label = self.remap_lut[semantic_label]

        sem_colors = self.sem_color_lut[semantic_label]
        if len(target_label_list) == 0:
            indices = np.arange(semantic_label.shape[0])
        else:
            indices = np.where(np.logical_not(np.isin(semantic_label, target_label_list)))[0]
            sem_colors[indices] = np.array(self.visual_cfg['background_points_color'])

        inst_label = label >> 16
        # inst_colors = self.__create_colors_from_inst_labels(inst_label)
        inst_colors = self.__create_colors_from_inst_labels_fixed(inst_label)

        # from copy import deepcopy
        # motion_label = deepcopy(semantic_label)
        motion_colors = np.zeros([semantic_label.shape[0], 3], dtype=np.float32)
        dynamic_ind = np.argwhere(semantic_label > 250).squeeze(1)
        static_ind = np.argwhere(semantic_label <= 250).squeeze(1)
        motion_colors[dynamic_ind] = np.array([1,0,0])
        motion_colors[static_ind] = np.array([1, 1, 1])

        self.sem_colors = sem_colors
        self.inst_colors = inst_colors
        self.motion_colors = motion_colors

        if self.color_mode == "sem":
            self.pcd.colors = o3d.utility.Vector3dVector(self.sem_colors)
        if self.color_mode == "inst":
            self.pcd.colors = o3d.utility.Vector3dVector(self.inst_colors)
        if self.color_mode == "motion":
            self.pcd.colors = o3d.utility.Vector3dVector(self.motion_colors)

    def __binary_color(self, labels, pos_color, neg_color):
        pos_ind = np.argwhere(labels == 1).squeeze(1)
        neg_ind = np.argwhere(labels == 0).squeeze(1)
        colors = np.zeros([labels.shape[0], 3], dtype=np.float32)
        colors[pos_ind] = np.array(pos_color)
        colors[neg_ind] = np.array(neg_color)
        return colors

    def __update_frame(self):
        if self.pcd_extension == '.bin':
            self.__load_pc_bin(self.pcd_dir[self.index])
        else:
            raise TypeError(f"point cloud data should end with in {self.pcd_extension}")
        if self.label_extension == '.label':
            self.__load_label_bin(self.label_dir[self.index])
        else:
            raise TypeError(f"label data should end with in {self.label_extension}")
        self.__print_msg()

    def __print_msg(self, msg: str = None):
        if msg is None:
            print(f"total : {self.index}/{self.end_frame - 1}")
        else:
            print(f"total : {self.index}/{self.end_frame - 1}, and {msg}")

    def __key_next_callback(self, vis):
        if self.index + 1 >= 0 and self.index + 1 < self.end_frame:
            self.index += self.interval
            self.__update_frame()
            self.vis.update_geometry(self.pcd)
            self.vis.update_renderer()
            self.vis.poll_events()
        else:
            self.__print_msg(f"no more point cloud!")

    def __key_back_callback(self, vis):
        if self.index - 1 >= 0 and self.index - 1 < self.end_frame:
            self.index -= self.interval
            self.__update_frame()
            self.vis.update_geometry(self.pcd)
            self.vis.update_renderer()
            self.vis.poll_events()
        else:
            self.__print_msg(f"no more point cloud!")

    def __key_sem_callback(self, vis):
        self.color_mode = "sem"
        # self.pcd.colors = o3d.utility.Vector3dVector(self.sem_colors)
        # self.vis.update_geometry(self.pcd)
        # self.vis.update_renderer()
        # self.vis.poll_events()
    def __key_inst_callback(self, vis):
        self.color_mode = "inst"
        # self.pcd.colors = o3d.utility.Vector3dVector(self.inst_colors)
        # self.vis.update_geometry(self.pcd)
        # self.vis.update_renderer()
        # self.vis.poll_events()

    def __key_motion_callback(self, vis):
        self.color_mode = "motion"

    def run(self):
        if self.pcd_dir is None:
            raise RuntimeError("Visualizer object should be set config via set_config_file(filename) before run!")
        self.vis.run()
        self.vis.destroy_window()

