import copy

import open3d as o3d
import numpy as np
import yaml
import json
import os
from O3DVisTools.kitti_utils import load_kitti_calib, read_objs2velo, colors_list
from O3DVisTools.open3d_geometry.open3d_box import create_box_from_dim_with_arrow
from O3DVisTools.open3d_geometry.open3d_coordinate import create_coordinate


usage = """
VisualizerKITTIObject Usage
Keyboard Options:
  'N'         Next frame
  'B'         Previous frame
"""


class VisualizerKITTIObject():

    def __init__(self, cfg_path):
        print(usage)
        with open(cfg_path, 'r') as cfg_file:
            visual_cfg = yaml.safe_load(cfg_file)
        self.data_path =  visual_cfg['data_root']
        self.visual_cfg = visual_cfg
        self.ctr_viewpoint_cfg = visual_cfg['ctr_viewpoint_cfg_path']

        lidar_path = os.path.join(self.data_path, 'velodyne')
        label_path = os.path.join(self.data_path, 'label_2')
        calib_path = os.path.join(self.data_path, 'calib')

        self.pcd_dir = [os.path.join(lidar_path, item)
                        for item in os.listdir(lidar_path)]
        self.pcd_dir.sort()
        self.label_dir = [os.path.join(label_path, item)
                        for item in os.listdir(label_path)]
        self.label_dir.sort()
        self.calib_dir = [os.path.join(calib_path, item)
                        for item in os.listdir(calib_path)]
        self.calib_dir.sort()

        self.visual_cfg = visual_cfg
        self.bool_inv = self.visual_cfg['bool_inv']
        self.target_label_list = self.visual_cfg['target_label_list']


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

        self.interval = visual_cfg['interval']
        self.index = visual_cfg['start_frame']
        if visual_cfg['end_frame'] == -1:
            self.end_frame = len(self.pcd_dir)
        else:
            self.end_frame = visual_cfg['end_frame']

        self.pcd = o3d.geometry.PointCloud()
        self.bboxes = None
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.vis.create_window(window_name='kitti-object', width=width, height=height, left=left, top=top,
                               visible=True)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.vis.add_geometry(mesh_frame)

        self.__update_frame()
        self.vis.add_geometry(self.pcd)
        for box in self.bboxes:
            self.vis.add_geometry(box)


        self.render_option = self.vis.get_render_option()
        self.render_option.background_color = np.asarray(self.visual_cfg['background_color'])
        self.render_option.point_size = point_size

        self.param = o3d.io.read_pinhole_camera_parameters(self.ctr_viewpoint_cfg)
        self.view_control = self.vis.get_view_control()
        self.view_control.set_front(front_vector)
        self.view_control.set_lookat(lookat_vector)
        self.view_control.set_up(up_vector)
        self.view_control.set_zoom(zoom_vector)
        # self.view_control.convert_from_pinhole_camera_parameters(self.param)

        self.vis.register_key_callback(ord('N'), self.__key_next_callback)
        self.vis.register_key_callback(ord('B'), self.__key_back_callback)

        print()

    def __load_pc_bin(self, pcd_path):
        points = np.fromfile(pcd_path, dtype=np.float32)
        points = points.reshape((-1, 4))
        coordinate = points[:, 0:3]
        colors = np.ones([coordinate.shape[0],3],dtype=np.float32)
        # remission = points[:, 3]
        self.pcd.points = o3d.utility.Vector3dVector(coordinate)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def __load_object_with_calib(self,calib_file,label_file):
        calib = load_kitti_calib(calib_file)
        boxes_velo, objs_type = read_objs2velo(label_file, calib['Tr_velo2cam'])
        box_colors = []
        if len(objs_type) == 0:
            box_colors = [[1, 0, 0] for i in range(boxes_velo.shape[0])]  # red
        else:
            box_colors = [colors_list[int(i)] for i in objs_type]

        boxes_o3d = []
        for i in range(boxes_velo.shape[0]):
            dim = boxes_velo[i]
            color = box_colors[i]
            box_o3d, arrow = create_box_from_dim_with_arrow(dim, color)
            boxes_o3d.append(box_o3d)
            # boxes_o3d.append(arrow)

        self.bboxes = boxes_o3d
    def __update_frame(self):
        self.__load_pc_bin(self.pcd_dir[self.index])
        self.__load_object_with_calib(self.calib_dir[self.index],self.label_dir[self.index])
        self.__print_msg()

    def __print_msg(self, msg: str = None):
        if msg is None:
            print(f"total : {self.index}/{self.end_frame - 1}")
        else:
            print(f"total : {self.index}/{self.end_frame - 1}, and {msg}")

    def __key_next_callback(self, vis):
        if self.index + 1 >= 0 and self.index + 1 < self.end_frame:
            self.index += self.interval
            self.vis.clear_geometries()
            self.__update_frame()
            self.vis.add_geometry(self.pcd)
            for box in self.bboxes:
                self.vis.add_geometry(box)
            # self.view_control.convert_from_pinhole_camera_parameters(self.param)
        else:
            self.__print_msg(f"no more point cloud!")

    def __key_back_callback(self, vis):
        if self.index - 1 >= 0 and self.index - 1 < self.end_frame:
            self.index -= self.interval
            self.vis.clear_geometries()
            self.__update_frame()
            self.vis.add_geometry(self.pcd)
            for box in self.bboxes:
                self.vis.add_geometry(box)
            # self.view_control.convert_from_pinhole_camera_parameters(self.param)
        else:
            self.__print_msg(f"Not exist last pcd!")

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


