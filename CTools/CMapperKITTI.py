import numpy as np
import open3d as o3d
import yaml
import gc
from CTools.CDataLoader import CDataLoader
from CTools.CFileProcesser import CFileProcesser
from CTools.CPoseProcesserKITTI import CPoseProcesserKITTI
from CTools.CVisualizer import CVisualizer
file_processer = CFileProcesser()
pose_processer = CPoseProcesserKITTI()
vis = CVisualizer()


class CMapperKITTI(CDataLoader):
    def mapper_with_T_xyzi(self,scan_files,Ts,start_frame,end_frame,bool_vis):
        points_map = np.empty([1, 4], dtype=np.float32)
        for i in range(start_frame,end_frame):
            print("%d / %d" % (i,end_frame))
            xyzi = self.load_bin_xyzi(scan_files[i])
            points,remissions = xyzi[:,0:3],xyzi[:,3]
            ones = np.ones(points.shape[0],dtype=np.float32)
            points_local = np.column_stack((points,ones))
            points_local = points_local.transpose()
            T = Ts[i]
            points_global = T@ points_local
            points_global = points_global.transpose()
            points_global[:,3] = remissions

            points_map = np.vstack((points_map,points_global))

        points_map = points_map[1:,:]
        points_map_to_vis = points_map[:,0:3]
        if bool_vis:
            vis.vis_cloud(points_map_to_vis)
        return points_map

    def mapper_with_T_xyz_label_color(self,scan_files,label_files,Ts,start_frame,end_frame,bool_vis):
        remap_lut, inv_lut, sem_color_lut = remapper.remaplut_invlut_colorlut()
        points_map = np.empty([1, 4], dtype=np.float32)
        labels_map = np.array([0],dtype=np.uint32)
        for i in range(start_frame,end_frame):
            print("%d / %d" % (i,end_frame))
            points,remissions = self.load_bin_xyzi(scan_files[i])
            labels = self.load_bin_label(label_files[i]) & 0xFFFF
            ones = np.ones(points.shape[0],dtype=np.float32)
            points_local = np.column_stack((points,ones))
            points_local = points_local.transpose()
            T = Ts[i]
            points_global = T@ points_local
            points_global = points_global.transpose()
            points_global[:,3] = remissions

            points_map = np.vstack((points_map,points_global))
            labels_map = np.hstack((labels_map,labels))

        points_map = points_map[1:,:]
        labels_map = labels_map[1:]
        colors_map = sem_color_lut[labels_map]

        points_map_to_vis = points_map[:,0:3]
        if bool_vis:
            # vis.vis_cloud_with_label(points_map_to_vis,colors_map,bg_color=[1,1,1])
            vis.vis_cloud_with_label(points_map_to_vis, colors_map)
            # vis.vis_cloud_with_label_white_background(points_map_to_vis,colors_map)
        return points_map,labels_map,colors_map

    def mapper_with_T_xyzlabel(self,scan_files,label_files,Ts,start_frame,end_frame,bool_vis):
        points_map = np.empty([1, 4], dtype=np.float32)
        labels_map = np.array([0],dtype=np.uint32)
        for i in range(start_frame,end_frame):
            print("%d / %d" % (i,end_frame))
            xyzi = self.load_bin_xyzi(scan_files[i])
            points,remissions = xyzi[:,0:3],xyzi[:,3]
            labels = self.load_bin_label(label_files[i]) & 0xFFFF
            ones = np.ones(points.shape[0],dtype=np.float32)
            points_local = np.column_stack((points,ones))
            points_local = points_local.transpose()
            T = Ts[i]
            points_global = T@ points_local
            points_global = points_global.transpose()
            points_global[:,3] = remissions

            points_map = np.vstack((points_map,points_global))
            labels_map = np.hstack((labels_map,labels))

        points_map = points_map[1:,:]
        labels_map = labels_map[1:]
        colors_map = vis.create_colors_from_labels(labels_map)

        points_map_to_vis = points_map[:,0:3]
        if bool_vis:
            # vis.vis_cloud_with_label(points_map_to_vis,colors_map,bg_color=[1,1,1])
            vis.vis_cloud_with_label(points_map_to_vis, colors_map)
            # vis.vis_cloud_with_label_white_background(points_map_to_vis,colors_map)
        return points_map,colors_map

    def mapper_with_T_xyzi_tensor(self,scan_files,label_files,Ts,start_frame,end_frame,
                                  bool_vis,vox_downsize,bool_save = False,
                                  pcd_name=None,label_name = None):
        # labels_dict, color_dict, color_dict_inv, contents_dict, learning_map, learning_map_inv, sem_color_lut = self.load_kitti_yaml_original()

        CFG = yaml.safe_load(open('/media/nio/Rambo/Data/KITTI-Semantic/config/semantic-kitti-new.yaml', 'r'))
        # color_dict = CFG["color_map"]
        # label_dict = CFG["labels"]

        points_map = np.empty([1, 4], dtype=np.float32)
        labels_map = np.array([0],dtype=np.uint32)
        for i in range(start_frame,end_frame):
            print("%d / %d" % (i,end_frame))
            points,remissions = self.load_bin_xyzi(scan_files[i])
            labels = self.load_bin_label(label_files[i]) & 0xFFFF
            ones = np.ones(points.shape[0],dtype=np.float32)
            points_local = np.column_stack((points,ones))
            points_local = points_local.transpose()
            T = Ts[i]
            points_global = T@ points_local
            points_global = points_global.transpose()
            points_global[:,3] = remissions

            points_map = np.vstack((points_map,points_global))
            labels_map = np.hstack((labels_map,labels))

        points_map = points_map[1:,0:3]
        labels_map = labels_map[1:]

        device = o3d.core.Device("CPU:0")
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point["positions"] = o3d.core.Tensor(points_map, o3d.core.float32, device)
        pcd.point["intensity"] = o3d.core.Tensor(labels_map, o3d.core.uint16, device)

        del points_map
        del labels_map

        downpcd = pcd.voxel_down_sample(voxel_size=vox_downsize)

        down_points = downpcd.point["positions"][:,:].numpy()
        down_labels = downpcd.point["intensity"][:].numpy()

        if bool_vis:
            vis.vis_cloud(down_points)
        if bool_save:
            pcd_to_be_save = o3d.geometry.PointCloud()
            pcd_to_be_save.points = o3d.utility.Vector3dVector(down_points)
            o3d.io.write_point_cloud(pcd_name,pcd_to_be_save)
            down_labels.tofile(label_name)

        return down_points,down_labels


