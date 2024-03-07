import numpy as np
import open3d as o3d
from CTools.CFileProcesser import CFileProcesser

class CPoseProcesserKITTI(CFileProcesser):

    def ToHomo(self,Rt):
        """
        将3x4的[R,t]转为4x4的
        [R,t
         0,1]
        """
        col = np.array([0, 0, 0, 1])
        T = np.vstack((Rt, col))
        return T

    def read_cam_poses(self,pose_file):
        """
        读取poses.txt到 (nx3x4)的相机位姿
        """
        poses = np.loadtxt(pose_file, dtype=np.float32).reshape(-1, 3, 4)
        return poses

    def cam_poses_to_homo(self,poses_rt):
        """
        将(nx3x4)的相机位姿转为(nx4x4)的相机位姿
        """
        poses_homo = np.zeros((poses_rt.shape[0], 4, 4))

        for i in range(poses_rt.shape[0]):
            pose_homo = poses_rt[i]
            pose_homo = self.ToHomo(pose_homo)
            poses_homo[i] = pose_homo
        return poses_homo

    def read_calib_projection(self,calib_file):
        """
        读取calib.txt到 P0, P1, P2, P3, Tr
        """
        # https://zhuanlan.zhihu.com/p/200696189
        calib = np.genfromtxt(calib_file)

        tmp_P0 = calib[0]
        P0 = tmp_P0[~np.isnan(tmp_P0)].reshape(3, 4)
        tmp_P1 = calib[1]
        P1 = tmp_P1[~np.isnan(tmp_P1)].reshape(3, 4)
        tmp_P2 = calib[2]
        P2 = tmp_P2[~np.isnan(tmp_P2)].reshape(3, 4)
        tmp_P3 = calib[3]
        P3 = tmp_P3[~np.isnan(tmp_P3)].reshape(3, 4)

        tmp_Tr = calib[4]
        Tr = tmp_Tr[~np.isnan(tmp_Tr)].reshape(3, 4)
        return P0, P1, P2, P3, Tr

    def read_pose_calib_to_Ts(self,pose_file,calib_file):
        """
        读取poses.txt、calib.txt并将其转换为 (nx4x4)的lidar pose
        """
        poses_cam = self.read_cam_poses(pose_file)
        P0, P1, P2, P3, T_cam_lidar = self.read_calib_projection(calib_file)
        T_cam_lidar = self.ToHomo(T_cam_lidar)
        T_lidar_cam = np.linalg.inv(T_cam_lidar)
        poses_cam_homo = self.cam_poses_to_homo(poses_cam)
        poses_lidar_homo = np.zeros((poses_cam.shape[0],4,4),dtype=np.float32)
        for i in range(poses_cam_homo.shape[0]):
            pose_cam = poses_cam_homo[i]
            pose_lidar = T_lidar_cam @ pose_cam @ T_cam_lidar
            poses_lidar_homo[i] = pose_lidar

        return poses_lidar_homo



if __name__ == '__main__':
    pos = CPoseProcesserKITTI()
    pose_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/00/poses.txt'
    calib_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/00/calib.txt'

    poses_cam = pos.read_cam_poses(pose_file)
    P0, P1, P2, P3, T_cam_lidar = pos.read_calib_projection(calib_file)
    poses_lidar_homo = pos.read_pose_calib_to_Ts(pose_file,calib_file)

    print(1)


