import os
import numpy as np
import open3d as o3d
import json
from CTools.CColor import *
import yaml

class CVisualizer():

    # [n,1]  ->  [n,3]
    # labels ->  colors
    def create_colors_from_labels(self,labels):
        np.random.seed(432)
        unique_labels = np.unique(labels)
        colors = np.zeros((len(labels), 3))
        for label in unique_labels:
            if label == -1:  # Ignore invalid label (if applicable)
                continue
            color = np.random.rand(3)  # Generate a random color for each instance
            mask = labels == label
            colors[mask] = color
        for_mask = np.argwhere(labels == 0).squeeze(1)
        colors[for_mask,:] = [1,1,1]
        return colors

    # [n,3] -> visualize
    # xyz可视化
    def vis_cloud(self,pc_xyz,point_size=1,color_list = None,background_color = [0,0,0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        if color_list is not None:
            pcd.paint_uniform_color(color_list)

        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()
        vis_.add_geometry(pcd)
        render_options = vis_.get_render_option()
        render_options.point_size = point_size
        render_options.background_color = np.array(background_color)
        vis_.run()

    # [n,3] + [n,3] -> visualize
    # xyz + colors 可视化
    def vis_cloud_with_label(self,pc_xyz,colors,voxDownsize=None,bg_color = [0,0,0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # print("before down sampled:", np.asarray(pcd.points).shape)
        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()

        if voxDownsize is None:
            pass
            # print("after down sampled:", np.asarray(pcd.points).shape)
            vis_.add_geometry(pcd)
        else:
            downpcd = pcd.voxel_down_sample(voxel_size=voxDownsize)
            # print("after down sampled:",np.asarray(downpcd.points).shape)
            vis_.add_geometry(downpcd)


        render_options = vis_.get_render_option()
        render_options.point_size = 2
        render_options.background_color = np.array(bg_color)
        vis_.run()


    # [n,3] + [n,3] -> visualize + image
    # xyz + colors 可视化且保存截屏到指定路径
    def vis_cloud_with_label_with_capture_viewpoint(self,pc_xyz,colors,image_save_path,viewpoint_path,voxDownsize=None,bg_color = [0,0,0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        test = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
        height = test.intrinsic.height
        width = test.intrinsic.width

        # print("before down sampled:", np.asarray(pcd.points).shape)
        vis_ = o3d.visualization.Visualizer()
        vis_.create_window(width=width,height=height)

        if voxDownsize is None:
            pass
            # print("after down sampled:", np.asarray(pcd.points).shape)
            vis_.add_geometry(pcd)
        else:
            downpcd = pcd.voxel_down_sample(voxel_size=voxDownsize)
            # print("after down sampled:",np.asarray(downpcd.points).shape)
            vis_.add_geometry(downpcd)


        render_options = vis_.get_render_option()
        render_options.point_size = 1
        render_options.background_color = np.array(bg_color)

        ctr = vis_.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
        vis_.update_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis_.poll_events()
        vis_.capture_screen_image(image_save_path)
        # vis_.run()
        import time
        time.sleep(1)
        vis_.destroy_window()


    # [n,3] + [n,3] + [pc1,pc2,...] + [color1,color2,...] -> visualize
    # 一部分点以逐点颜色形式出现
    # 另一部分点为点云列表+颜色列表出现
    def vis_cloud_with_label_plus_list(self,pc_xyz,colors,pc_list,color_list,voxDownsize=None,bg_color = [0,0,0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # print("before down sampled:", np.asarray(pcd.points).shape)
        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()

        if voxDownsize is None:
            pass
            # print("after down sampled:", np.asarray(pcd.points).shape)
            vis_.add_geometry(pcd)
        else:
            downpcd = pcd.voxel_down_sample(voxel_size=voxDownsize)
            # print("after down sampled:",np.asarray(downpcd.points).shape)
            vis_.add_geometry(downpcd)

        if len(pc_list) != len(color_list):
            print("pc_list must equal color_list!")
        else:
            for i in range(len(pc_list)):
                pc = pc_list[i]
                color_pc = color_list[i]
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(pc)
                cloud.paint_uniform_color(color_pc)
                vis_.add_geometry(cloud)


        render_options = vis_.get_render_option()
        render_options.point_size = 2
        render_options.background_color = np.array(bg_color)
        vis_.run()

    # [n,3] + [n,3] + [pc1,pc2,...] + [color1,color2,...] -> visualize + image
    # 一部分点以逐点颜色形式出现
    # 另一部分点为点云列表+颜色列表出现
    # 截屏
    def vis_cloud_with_label_plus_list_with_capture_viewpoint(self,pc_xyz,colors,pc_list,color_list,image_save_path,viewpoint_path,voxDownsize=None,bg_color = [0,0,0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        test = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
        height = test.intrinsic.height
        width = test.intrinsic.width

        # print("before down sampled:", np.asarray(pcd.points).shape)
        vis_ = o3d.visualization.Visualizer()
        vis_.create_window(width=width,height=height)

        if voxDownsize is None:
            pass
            # print("after down sampled:", np.asarray(pcd.points).shape)
            vis_.add_geometry(pcd)
        else:
            downpcd = pcd.voxel_down_sample(voxel_size=voxDownsize)
            # print("after down sampled:",np.asarray(downpcd.points).shape)
            vis_.add_geometry(downpcd)

        if len(pc_list) != len(color_list):
            print("pc_list must equal color_list!")
        else:
            o3d_list = []
            for i in range(len(pc_list)):
                pc = pc_list[i]
                color_pc = color_list[i]
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(pc)
                cloud.paint_uniform_color(color_pc)
                o3d_list.append(cloud)
                vis_.add_geometry(cloud)

        render_options = vis_.get_render_option()
        render_options.point_size = 1
        render_options.background_color = np.array(bg_color)

        ctr = vis_.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
        for geo in o3d_list:
            vis_.update_geometry(geo)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis_.poll_events()
        vis_.capture_screen_image(image_save_path)
        # vis_.run()
        import time
        time.sleep(1)
        vis_.destroy_window()

    # [n,3] + [n,3] -> visualize
    # xyz + xyz 可视化
    # 用来对比两个点云
    def vis_two_cloud(self,pc1,pc2,color_pc1 = [1, 1, 1],color_pc2 = [1, 0, 0]):

        cloud1 = o3d.geometry.PointCloud()
        cloud1.points = o3d.utility.Vector3dVector(pc1)
        cloud1.paint_uniform_color(color_pc1)

        cloud2 = o3d.geometry.PointCloud()
        cloud2.points = o3d.utility.Vector3dVector(pc2)
        cloud2.paint_uniform_color(color_pc2)

        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()
        vis_.add_geometry(cloud1)
        vis_.add_geometry(cloud2)
        render_options = vis_.get_render_option()
        render_options.point_size = 3
        render_options.background_color = np.array([0, 0, 0])
        # render_options.background_color = np.array([1, 1, 1])
        vis_.run()

    # [pc1,pc2,...] + [color1,color2,...] -> visualize
    # 可视化多个点云，每个点云对应到不同的颜色
    def vis_n_cloud(self,pc_list,color_list,bg_color=[1,1,1]):

        if len(pc_list) != len(color_list):
            print("pc_list must equal color_list!")
        else:
            vis_ = o3d.visualization.Visualizer()
            vis_.create_window()
            for i in range(len(pc_list)):
                pc = pc_list[i]
                color_pc = color_list[i]
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(pc)
                cloud.paint_uniform_color(color_pc)
                vis_.add_geometry(cloud)
            render_options = vis_.get_render_option()
            render_options.point_size = 1
            render_options.background_color = np.array(bg_color)
            vis_.run()

    # [n,3] + [n,1] -> visualize
    # 根据label创建点云的颜色并可视化
    def vis_pts_wrt_label(self,points, labels):
        np.random.seed(42)
        pts_list = []
        color_list = []
        for tmp in np.unique(labels):
            ind = np.argwhere(labels == tmp).squeeze(1)
            pts = points[ind]
            color = np.random.rand(3)
            pts_list.append(pts)
            color_list.append(color)

        # self.vis_n_cloud(pts_list, color_list, bg_color=BLACK)
        self.vis_n_cloud(pts_list, color_list, bg_color=WHITE)

    # [pc1,pc2,...] + [color1,color2,...] + [obb1,obb2,...]
    # 一帧点云，不同的点颜色不同，且有定向包围框可视化
    def vis_n_cloud_with_obb(self,pc_list,color_list,obb_list,bg_color=[1,1,1]):

        if len(pc_list) != len(color_list):
            print("pc_list must equal color_list!")
        else:
            vis_ = o3d.visualization.Visualizer()
            vis_.create_window()
            for i in range(len(pc_list)):
                pc = pc_list[i]
                color_pc = color_list[i]
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(pc)
                cloud.paint_uniform_color(color_pc)
                vis_.add_geometry(cloud)

            for i in range(len(obb_list)):
                obb = obb_list[i]
                vis_.add_geometry(obb)

            render_options = vis_.get_render_option()
            render_options.point_size = 1
            render_options.background_color = np.array(bg_color)
            vis_.run()

    # [pc1,pc2,...] + [color1,color2,...] -> visualize
    # 可视化多个点云，每个点云对应到不同的颜色
    # 且保存截屏
    def vis_n_cloud_with_capture_viewpoint(self,pc_list,color_list,image_save_path,viewpoint_path,bg_color=[1,1,1]):
        test = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
        height = test.intrinsic.height
        width = test.intrinsic.width
        if len(pc_list) != len(color_list):
            print("pc_list must equal color_list!")
        else:
            vis_ = o3d.visualization.Visualizer()
            vis_.create_window(width=width,height=height)
            o3d_list = []
            for i in range(len(pc_list)):
                pc = pc_list[i]
                color_pc = color_list[i]
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(pc)
                cloud.paint_uniform_color(color_pc)
                o3d_list.append(cloud)
                vis_.add_geometry(cloud)
            render_options = vis_.get_render_option()
            render_options.point_size = 1
            render_options.background_color = np.array(bg_color)

            ctr = vis_.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
            for geo in o3d_list:
                vis_.update_geometry(geo)
            ctr.convert_from_pinhole_camera_parameters(param)
            vis_.poll_events()
            vis_.capture_screen_image(image_save_path)
            # vis_.run()
            import time
            time.sleep(1)
            vis_.destroy_window()

    # [n,3] + [1,1] -> visualize
    # 可视化带法向量
    def vis_cloud_with_normal(self,pc_xyz,pt_color,search_radius,search_max_nn,voxDownsize=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        print("before down sampled:", np.asarray(pcd.points).shape)
        pcd.paint_uniform_color(pt_color)
        if voxDownsize is None:
            pass
            print("after down sampled:", np.asarray(pcd.points).shape)
        else:
            pcd = pcd.voxel_down_sample(voxel_size=voxDownsize)
            print("after down sampled:",np.asarray(pcd.points).shape)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius,max_nn=search_max_nn))
        normals = np.array(pcd.normals,dtype=np.float32)
        o3d.visualization.draw_geometries([pcd],window_name="point cloud with normal",
                                          point_show_normal=True)
        return normals





