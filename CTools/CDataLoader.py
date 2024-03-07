import numpy as np
import open3d as o3d
from CTools.CVisualizer import CVisualizer
vis = CVisualizer()

class CDataLoader():
    def load_bin_xyz(self,bin_file):
        xyz = np.fromfile(bin_file,dtype=np.float32).reshape((-1,4))[:,0:3]
        return xyz

    def load_bin_xyzi(self,bin_file):
        xyzi = np.fromfile(bin_file,dtype=np.float32).reshape((-1,4))
        return xyzi

    def load_bin_xyzi_crop(self,bin_file,roi):
        xyzi = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
        print("original points number(without remissions): ", xyzi.shape)
        x = xyzi[:, 0]
        y = xyzi[:, 1]
        z = xyzi[:, 2]
        dist_to_origin = x * x + y * y
        dist_to_origin_ind = np.argwhere(dist_to_origin < roi * roi).squeeze(1)
        # print("outlier number: ", dist_to_origin_ind[0].shape)
        new_xyzi = xyzi[dist_to_origin_ind]
        return new_xyzi

    def load_bin_label(self,label_file):
        label = np.fromfile(label_file,dtype=np.uint32).reshape((-1))
        return label

    def vis_bin_xyz(self,bin_file):
        xyz = self.load_bin_xyz(bin_file)
        vis.vis_cloud(xyz)


    def load_pcd_xyz(self,pcd_file):
        xyz =o3d.io.read_point_cloud(pcd_file,remove_nan_points=True)
        xyz = np.asarray(xyz.points,dtype=np.float32)
        return xyz

    def load_pcd_xyz_downsize(self,pcd_file,downsize):
        pcd = o3d.io.read_point_cloud(pcd_file,remove_nan_points=True)
        ds_pcd = pcd.voxel_down_sample(voxel_size=downsize)
        xyz = np.asarray(ds_pcd.points,dtype=np.float32)
        return xyz

    def load_pcd_xyzi(self,pcd_file):
        pcd = o3d.t.io.read_point_cloud(pcd_file)
        i = pcd.point["intensity"]  # 强度
        xyz = pcd.point["positions"]  # 坐标
        i = i[:, :].numpy()  # 转换为数组类型
        xyz = xyz[:, :].numpy()  # 转换为数组类型
        nan_ind = np.argwhere(np.isnan(xyz[:,0])).squeeze(1)
        all_ind = np.arange(0,xyz.shape[0])
        not_nan_ind = np.setdiff1d(all_ind,nan_ind)
        xyz = xyz[not_nan_ind]
        i = i[not_nan_ind]
        xyzi = np.column_stack((xyz,i))
        return xyzi
    def load_pcd_xyzi_downsize(self,pcd_file,downsize):
        pcd = o3d.t.io.read_point_cloud(pcd_file)
        ds_pcd = pcd.voxel_down_sample(voxel_size=downsize)
        pcd = ds_pcd
        xyz = pcd.point["positions"].numpy().astype(np.float32)
        i = pcd.point["intensity"].numpy().astype(np.uint32).reshape((-1))
        xyzi = np.column_stack((xyz,i))
        return xyzi
    def load_pcd_xyzrgb(self,pcd_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        xyz = np.asarray(pcd.points,dtype=np.float32)
        rgb = np.asarray(pcd.colors,dtype=np.float32)
        return xyz,rgb


    def vis_pcd_xyz(self,pcd_file):
        points = self.load_pcd_xyz(pcd_file)
        vis.vis_cloud(points)

    def wrap_xyz_to_o3d(self,xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd

    def wrap_xyzi_to_o3d(self,xyzi):
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point["positions"] = o3d.core.Tensor(xyzi[:,0:3], dtype, device)
        pcd.point["intensity"] = o3d.core.Tensor(xyzi[:,3], dtype, device)
        return pcd
