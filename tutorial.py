import numpy as np
import cv2
import open3d as o3d
from CTools.CVisualizerKITTIObject import VisualizerKITTIObject
from CTools.CVisualizerKITTISemInstMotion import VisualizerKITTISemInst
from CTools.CMapperKITTI import CMapperKITTI
from CTools.CDataLoader import CDataLoader
from CTools.CVisualizer import CVisualizer
from CTools.CBEVer import CBEVer
from CTools.CFileProcesser import CFileProcesser
from CTools.CPoseProcesserKITTI import CPoseProcesserKITTI
from CTools.CColor import *


def vis_sequential_object():
    visualizer_object = VisualizerKITTIObject("./config/_VisConfigObj/kitti_mini_object.yaml")
    visualizer_object.run()
def vis_sequential_sem_inst_motion():
    visualizer_sem_inst_motion = VisualizerKITTISemInst("./config/_VisConfigSemInstMotion/gt-kitti.yaml")
    visualizer_sem_inst_motion.run()

def test_visualizer():
    loader = CDataLoader()
    vis = CVisualizer()
    bin_file = "./data/KITTI-Semantic/sequences/00/velodyne/000000.bin"
    label_file = "./data/KITTI-Semantic/sequences/00/labels/000000.label"

    bin_file2 = "./data/KITTI-Semantic/sequences/00/velodyne/000005.bin"
    label_file2 = "./data/KITTI-Semantic/sequences/00/labels/000005.label"

    xyz = loader.load_bin_xyz(bin_file)
    xyz2 = loader.load_bin_xyz(bin_file2)
    label = loader.load_bin_label(label_file)
    label_sem = label & 0xFFFF
    label_inst = label >> 16
    color_sem = vis.create_colors_from_labels(label_sem)

    print("vis.vis_two_cloud(xyz,xyz2)")
    vis.vis_two_cloud(xyz,xyz2)

    print("color_inst = vis.create_colors_from_labels(label_inst)")
    color_inst = vis.create_colors_from_labels(label_inst)

    print("vis.vis_cloud(xyz)")
    vis.vis_cloud(xyz)
    print("vis.vis_cloud_with_label(xyz,color_sem)")
    vis.vis_cloud_with_label(xyz,color_sem)
    print("vis.vis_cloud_with_label(xyz,color_inst)")
    vis.vis_cloud_with_label(xyz,color_inst)

    print("vis.vis_pts_wrt_label(xyz, label_inst)")
    vis.vis_pts_wrt_label(xyz, label_inst)
    print("vis.vis_pts_wrt_label(xyz, label_sem)")
    vis.vis_pts_wrt_label(xyz, label_sem)

    pts_list = []
    color_list = []
    for l in np.unique(label_sem):
        ind = np.argwhere(label_sem == l).squeeze(1)
        pts = xyz[ind]
        color = np.random.rand(3)
        color_list.append(color)
        pts_list.append(pts)

    print("vis.vis_n_cloud(pts_list,color_list,bg_color=WHITE)")
    vis.vis_n_cloud(pts_list,color_list,bg_color=WHITE)


def test_dataloader_visualizer():
    loader = CDataLoader()
    vis = CVisualizer()
    ROI = 20 # unit:m
    downsize = 0.2

    bin_file = "./data/KITTI-Semantic/sequences/00/velodyne/000000.bin"
    label_file = "./data/KITTI-Semantic/sequences/00/labels/000000.label"
    pcd_file = "./data/Sample-pcd/000000.pcd"

    bin_xyz = loader.load_bin_xyz(bin_file=bin_file)
    bin_xyzi = loader.load_bin_xyzi(bin_file=bin_file)
    bin_xyzi_crop = loader.load_bin_xyzi_crop(bin_file=bin_file,roi=ROI)
    bin_label = loader.load_bin_label(label_file=label_file)
    pcd_xyzi = loader.load_pcd_xyzi(pcd_file=pcd_file)
    pcd_xyzi_downsize = loader.load_pcd_xyzi_downsize(pcd_file=pcd_file,downsize=downsize)

    print("bin_xyz [%d,%d]" % (bin_xyz.shape[0],bin_xyz.shape[1]))
    print("bin_xyzi [%d,%d]" % (bin_xyzi.shape[0], bin_xyzi.shape[1]))
    print("bin_label [%d]" % (bin_label.shape[0]))
    print("pcd_xyzi [%d,%d]" % (pcd_xyzi.shape[0], pcd_xyzi.shape[1]))

    print("vis bin_xyz")
    vis.vis_cloud(bin_xyz)
    print("vis bin_xyzi[:,0:3]")
    vis.vis_cloud(bin_xyzi[:,0:3])
    print("vis bin_xyzi[:,0:3], ROI = 50m")
    vis.vis_cloud(bin_xyzi_crop[:,0:3])
    print("vis bin_xyz + bin_label")
    vis.vis_pts_wrt_label(bin_xyz, bin_label)

    print("vis pcd_xyzi[:,0:3]")
    vis.vis_cloud(pcd_xyzi[:,0:3])
    print("vis pcd_xyzi[:,0:3] + pcd_xyzi[:,3] & 0xFFFF")
    vis.vis_pts_wrt_label(pcd_xyzi[:,0:3], pcd_xyzi[:,3].astype(np.uint32)&0xFFFF)
    print("vis pcd_xyzi_downsize[:,0:3]")
    vis.vis_cloud(pcd_xyzi_downsize[:,0:3])
    print("compare pcd_xyzi & pcd_xyzi_downsize")
    vis.vis_two_cloud(pcd_xyzi[:,0:3],pcd_xyzi_downsize[:,0:3])

    loader.vis_bin_xyz(bin_file)
    loader.vis_pcd_xyz(pcd_file)

def test_kitti_mapper():
    mapper = CMapperKITTI()
    vis = CVisualizer()
    file_processer = CFileProcesser()
    pose_processer = CPoseProcesserKITTI()
    bin_path = './data/KITTI-Semantic/sequences/00/velodyne'
    label_path = './data/KITTI-Semantic/sequences/00/labels'
    pose_file = './data/KITTI-Semantic/sequences/00/poses.txt'
    calib_file = './data/KITTI-Semantic/sequences/00/calib.txt'

    bin_files = file_processer.get_filenames(bin_path)
    label_files = file_processer.get_filenames(label_path)
    poses_lidar_homo = pose_processer.read_pose_calib_to_Ts(pose_file, calib_file)

    start_frame = 0
    end_frame = 5
    bool_vis = True

    map = mapper.mapper_with_T_xyzi(bin_files,poses_lidar_homo,start_frame,end_frame,bool_vis)
    xyzi, rgb = mapper.mapper_with_T_xyzlabel(bin_files, label_files, poses_lidar_homo, start_frame, end_frame,
                                              bool_vis)
    vis.vis_cloud_with_label(xyzi[:, 0:3], rgb)

def test_bev():

    xyz_file = "./data/KITTI-Semantic/sequences/00/velodyne/000005.bin"
    label_file = "./data/KITTI-Semantic/sequences/00/labels/000005.label"
    points = np.fromfile(xyz_file,dtype=np.float32).reshape((-1,4))
    label = np.fromfile(label_file,dtype=np.uint32).reshape((-1))
    sem = label & 0xFFFF
    inst = label >> 16


    x_min = -30
    x_max = 30
    y_min = -30
    y_max = 30
    z_min = -2.73
    z_max = 1.27
    bound = [x_min,x_max,y_min,y_max,z_min,z_max]

    bever = CBEVer(points,sem=sem,inst=inst,bound=bound)
    bever.extract_inlier()
    height_map = bever.xyz_to_bev()
    cv2.imshow("bev_map", height_map)
    cv2.waitKey(-1)

    height_map, sem_color_map, inst_color_map = bever.xyz_sem_inst_to_bev()
    cv2.imshow("bev_map", height_map)
    cv2.waitKey(-1)

    cv2.imshow("bev_map", sem_color_map)
    cv2.waitKey(-1)

    cv2.imshow("bev_map", inst_color_map)
    cv2.waitKey(-1)

    combined = bever.combine_img_1x3(height_map,sem_color_map,inst_color_map)
    cv2.imshow("bev_map", combined)
    cv2.waitKey(-1)

if __name__ == '__main__':
    # print("sample: visualize sequential point cloud with object OBBs")
    # vis_sequential_object()

    # print("sample: visualize sequential point cloud with Semantic/Instance/Motion labels")
    # vis_sequential_sem_inst_motion()

    # print("sample: visualize single frame point cloud")
    # test_visualizer()

    # print("sample: load point cloud")
    # test_dataloader_visualizer()

    # print("sample: kitti mapper")
    # test_kitti_mapper()

    print("sample: bev")
    test_bev()
