import numpy as np
import open3d as o3d
import argparse
import cv2

from scipy.spatial.transform import Rotation as R
from typing import List
from typing import Dict

import os
from tqdm import tqdm


def cos_depth_map(K, width, height):

    uv_map = np.array([[[i, j, 1] for i in range(width)] for j in range(height)])
    uv_map = uv_map[..., np.newaxis]

    inv_K = np.linalg.inv(K)
    inv_K = inv_K.reshape(1, 1, 3, 3)

    xy_hom_map = inv_K @ uv_map
    norm_map = np.linalg.norm(xy_hom_map[:,:,:2, -1], axis=-1)

    cos_map = 1. / norm_map

    return cos_map

def process_depth_anything(K, width, height, depth):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - cx) / fx
    y = (y - cy) / fy




def get_name_postfix_type(path):

    name_postfix = os.path.basename(path).split('.')[0]
    ext = os.path.basename(path).split('.')[1]

    if "_" in name_postfix:
        name = name_postfix.split('.')[0].split('_')[0]
        postfix = name_postfix.split('.')[0].split('_')[1]
    else:
        name = name_postfix
        postfix = ""

    return name, postfix, ext
def split_name(path):
    return get_name_postfix_type(path)[0]

def split_index(path):
    return int(split_name(path))


def visualize_depth_image_with_open3d(depth_path, color_path, fx, fy, cx, cy, pose_file = None, bgr2rgb=True, merged_view=False, start_idx=-1, end_idx=-1, scale_factor=1,
                                    depth_postfix="", color_postfix="", pcd_save_path=None):
    fx = fx / scale_factor
    fy = fy / scale_factor
    cx = cx / scale_factor
    cy = cy / scale_factor

    depth_paths = []
    color_paths = []

    def is_rgb_image(path):
        _, postfix, ext = get_name_postfix_type(path)
        return (postfix == color_postfix and ext == "jpg") or (postfix == color_postfix and ext == "png")


    def is_depth_image(path):
        _, postfix, ext = get_name_postfix_type(path)
        return (postfix == depth_postfix and ext == "npy")

    if os.path.isdir(depth_path) and os.path.isdir(color_path):
        assert(os.path.isdir(depth_path) and os.path.isdir(color_path)), "The depth and color image paths should be directories"

        depth_paths = [os.path.join(depth_path, f) for f in os.listdir(depth_path) if is_depth_image(f)]
        color_paths = [os.path.join(color_path, f) for f in os.listdir(color_path) if is_rgb_image(f)]

        depth_paths = sorted(depth_paths)
        color_paths = sorted(color_paths)


        if start_idx != -1 and end_idx != -1:
            depth_paths = [d for d in depth_paths if split_index(d) >= start_idx and split_index(d) <= end_idx]
            color_paths = [c for c in color_paths if split_index(c) >= start_idx and split_index(c) <= end_idx]

        shared_depth_color_images = list(set([split_name(d) for d in depth_paths]) & set([split_name(c) for c in color_paths]))

        depth_paths = [d for d in depth_paths if split_name(d) in shared_depth_color_images]
        color_paths = [c for c in color_paths if split_name(c) in shared_depth_color_images]

        # shared_depth_color_images = list(set([os.path.basename(d).split('.')[0] for d in depth_paths]) & set([os.path.basename(c).split('.')[0] for c in color_paths]))
        # depth_paths = [d for d in depth_paths if os.path.basename(d).split('.')[0] in shared_depth_color_images]
        # color_paths = [c for c in color_paths if os.path.basename(c).split('.')[0] in shared_depth_color_images]

        assert len(depth_paths) > 0, "No depth images found in the directory"
        assert len(color_paths) > 0, "No color images found in the directory"
        assert len(depth_paths) == len(color_paths), "The number of depth images and color images should be the same : {} != {}".format(len(depth_paths), len(color_paths))
        assert all([split_name(d) == split_name(c) for d, c in zip(depth_paths, color_paths)]), "Depth and color images should have the same name"
        # assert all([os.path.basename(d).split('.')[0] == os.path.basename(c).split('.')[0] for d, c in zip(depth_paths, color_paths)]), "Depth and color images should have the same name"
    else:
        merged_view = False
        depth_paths.append(depth_path)
        color_paths.append(color_path)
        start_idx = -1
        end_idx = -1
        print("Merged view is toogle to False because the input is not a directory")
        print("Unset start and end index because the input is not a directory")

    color_depth_paths = list(zip(color_paths, depth_paths))



    poses = {}
    if merged_view == True:
        assert pose_file is not None, "Poses file should be provided for merged view"
        if os.path.dirname(pose_file) != os.path.basename(depth_path):
            print("Warning : The pose file is not in the same directory as the depth images")

        print("Pose File: ", pose_file)

        with open(pose_file, 'r') as f:
            lines = [line for line in f.readlines() if len(line) > 0 and not line.startswith('#')]
            for line in lines:
                if start_idx != -1 and end_idx != -1:
                    image_id = int(line.split()[0])
                    if image_id < start_idx or image_id > end_idx:
                        continue

                image_id = int(line.split()[0])
                T00, T01, T02, T03, T10, T11, T12, T13, T20, T21, T22, T23, T30, T31, T32, T33 = list(map(float, line.split()[1:]))
                T_wc = np.array([[T00, T01, T02, T03],
                                [T10, T11, T12, T13],
                                [T20, T21, T22, T23],
                                [T30, T31, T32, T33]])
                poses[image_id] = np.linalg.inv(T_wc)

        # assert len(poses) == len(color_depth_paths), "The number of poses should be the same as the number of images"

    print("Number of images: ", len(depth_paths))

    color_depth_images_and_ids = []

    resize_flag = False

    for color_path, depth_path in tqdm(color_depth_paths, desc="Loading images"):
        image_id = split_index(depth_path)
        if start_idx != -1 and end_idx != -1:
            if image_id < start_idx or image_id > end_idx:
                continue

        depth_image = np.load(depth_path)

        color_image = cv2.imread(color_path)
        if bgr2rgb == True:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        if depth_image.ndim > 2:
            print("Depth image should be a single channel image")
        if color_image.ndim != 3:
            print("Color image should be a 3 channel image")
        if depth_image.shape[:2] != color_image.shape[:2]:
            if resize_flag == False:
                resize_flag = True
                fx = fx * depth_image.shape[1] / color_image.shape[1]
                fy = fy * depth_image.shape[0] / color_image.shape[0]
                cx = cx * depth_image.shape[1] / color_image.shape[1]
                cy = cy * depth_image.shape[0] / color_image.shape[0]
            color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_LINEAR) 

        depth_image = 255.0 - depth_image


        print(fx, fy, cx, cy)
        print(color_image.shape, depth_image.shape)
        color_depth_images_and_ids.append((color_image, depth_image, image_id))

    pcds = None
    for color_image, depth_image, image_id in tqdm(color_depth_images_and_ids, desc="Creating point clouds"):

        # Image dimensions and camera intrinsics
        width, height = depth_image.shape

        # Create Open3D camera intrinsic object
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width, height=height,
                                                       fx=fx, fy=fy, cx=cx, cy=cy)

        extrinsics = poses[image_id] if merged_view == True else np.eye(4)

        # Convert the depth image to an Open3D depth image
        o3d_depth_image = o3d.geometry.Image(depth_image.astype(np.float32))

        # Convert the color image to an Open3D color image
        o3d_color_image = o3d.geometry.Image(color_image)

        # Create Open3D RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color_image, o3d_depth_image, depth_scale=1.0, depth_trunc=255.0, convert_rgb_to_intensity=False)
        
        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics, extrinsics)
 
        if merged_view == False:
            # Visualize the point cloud
            print("Visualizing image and depth file : ", color_path, depth_path)
            print("Number of points: ", np.asarray(pcd.points).shape[0])
            o3d.visualization.draw_geometries([pcd])
        else:
            # pcd = pcd.random_down_sample(0.003)
            if pcds is None:
                pcds = pcd
            else:
                pcds += pcd

    num_point_total = np.asarray(pcds.points).shape[0]
    print("Total number of points: ", num_point_total)

    if merged_view == True:
        # Visualize the point cloud
        print("visualizing all the point clouds in a single view")
        o3d.visualization.draw_geometries([pcds])
        if pcd_save_path is not None:
            o3d.io.write_point_cloud(pcd_save_path, pcds)

def visualize_sparse_rgbd_data_with_open3d(data_path, fx, fy, cx, cy, pose_file = None, bgr2rgb=True, merged_view=False, start_idx=-1, end_idx=-1, scale_factor=1):

    if os.path.isdir(data_path):

        data_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]

        if start_idx != -1 and end_idx != -1:
            data_paths = [d for d in data_paths if int(os.path.basename(d).split('.')[0]) >= start_idx and int(os.path.basename(d).split('.')[0]) <= end_idx]


        assert len(data_paths) > 0, "No depth images found in the directory"
    else:
        merged_view = False
        data_paths =[data_path]
        start_idx = -1
        end_idx = -1
        print("Merged view is toogle to False because the input is not a directory")
        print("Unset start and end index because the input is not a directory")


    poses = {}
    if merged_view == True:
        assert pose_file is not None, "Poses file should be provided for merged view"
        assert(os.path.dirname(pose_file) == os.path.basename(data_path)), "The pose file should be in the same directory as the depth images"

        print("Pose File: ", pose_file)

        with open(pose_file, 'r') as f:
            lines = [line for line in f.readlines() if len(line) > 0 and not line.startswith('#')]
            for line in lines:
                if start_idx != -1 and end_idx != -1:
                    image_id = int(line.split()[0])
                    if image_id < start_idx or image_id > end_idx:
                        continue

                image_id = int(line.split()[0])
                T00, T01, T02, T03, T10, T11, T12, T13, T20, T21, T22, T23, T30, T31, T32, T33 = list(map(float, line.split()[1:]))
                T_wc = np.array([[T00, T01, T02, T03],
                                [T10, T11, T12, T13],
                                [T20, T21, T22, T23],
                                [T30, T31, T32, T33]])
                poses[image_id] = T_wc

        # assert len(poses) == len(color_depth_paths), "The number of poses should be the same as the number of images"

    print("Number of images: ", len(data_paths))

    data_mat_and_ids = []
    for data_path in tqdm(data_paths, desc="Loading images"):
        image_id = int(os.path.basename(data_path).split('.')[0])
        if start_idx != -1 and end_idx != -1:
            if image_id < start_idx or image_id > end_idx:
                continue

        data_mat = np.load(data_path)


        data_mat_and_ids.append((data_mat, image_id)) 

    pcds = []
    for data_mat, image_id in tqdm(data_mat_and_ids, desc="Creating point clouds"):

        pcd = o3d.geometry.PointCloud()


        extrinsics = poses[image_id] if merged_view == True else np.eye(4)

        num_data = data_mat.shape[0]

        if num_data != 0:

            colors = data_mat[:,3:6] / 255.0
            xy = data_mat[:,0:2]
            d = data_mat[:,2,np.newaxis]
            ones = np.ones((num_data, 1))
            pts_c = np.hstack((xy*d, d, ones))

            T_wc = poses[image_id]
            pts_w = (T_wc @ pts_c.transpose()).transpose()

            points = pts_w[:, :3]

            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        if merged_view == False:
            # Visualize the point cloud
            print("Number of points: ", np.asarray(pcd.points).shape[0])
            o3d.visualization.draw_geometries([pcd])
        else:
            pcds.append(pcd)

    num_point_total = 0
    for pcd in pcds:
        num_point_total += np.asarray(pcd.points).shape[0]
    print("Total number of points: ", num_point_total)
    print(merged_view)

    if merged_view == True:
        # Visualize the point cloud
        print("visualizing all the point clouds in a single view")
        o3d.visualization.draw_geometries(pcds)


def main():
    parser = argparse.ArgumentParser(description="Visualize a depth image as a point cloud using Open3D.")
    parser.add_argument("--depth_image", help="Path to the .npy depth image file or directory")
    parser.add_argument("--color_image", help="Path to the .jpg/png color image file or directory")
    parser.add_argument("--sparse_rgbd_data", default=None, help="Path to the .npy sparse rgbd file or directory")
    parser.add_argument("--model_path", default = None, help="Path to the colmap model directory")
    parser.add_argument("--fx", type=float, default=2981.5513, help="Focal length in x direction")
    parser.add_argument("--fy", type=float, default=2981.5513, help="Focal length in y direction")
    parser.add_argument("--cx", type=float, default=2304.0, help="Principal point x coordinate")
    parser.add_argument("--cy", type=float, default=1728.0, help="Principal point y coordinate")    
    parser.add_argument("--scale_factor", type=int, default=1, help="Division scale factor that will be applied to the given intrinsic parameters")
    parser.add_argument("--bgr2rgb", default="True", help="Convert BGR to RGB")
    parser.add_argument("--pose_file", default=None, help="Path to the pose file")
    parser.add_argument("--merged_view", default="False", help="Visualize all the point clouds in a single view")
    parser.add_argument("--start_idx", type=int, default=-1, help="Start index when visualizing multiple images")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index when visualizing multiple images")
    parser.add_argument("--depth_postfix", default="", help="Depth image postfix")
    parser.add_argument("--color_postfix", default="", help="Color image postfix")
    parser.add_argument("--pcd_save_path", default=None, help="Path to save the point cloud")
    args = parser.parse_args()

    def StringToBool(s):
        if s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        else:
            raise ValueError
    bgr2rgb = StringToBool(args.bgr2rgb)
    merged_view = StringToBool(args.merged_view)

    if args.sparse_rgbd_data == None:
        visualize_depth_image_with_open3d(args.depth_image, args.color_image, args.fx, args.fy, args.cx, args.cy,
                                          pose_file=args.pose_file, bgr2rgb=bgr2rgb, merged_view=merged_view,
                                          start_idx=args.start_idx, end_idx=args.end_idx, scale_factor=args.scale_factor,
                                        color_postfix=args.color_postfix, depth_postfix=args.depth_postfix, pcd_save_path=args.pcd_save_path)
    else:
        visualize_sparse_rgbd_data_with_open3d(args.sparse_rgbd_data, args.fx, args.fy, args.cx, args.cy,
                                          pose_file=args.pose_file, bgr2rgb=bgr2rgb, merged_view=merged_view,
                                          start_idx=args.start_idx, end_idx=args.end_idx, scale_factor=args.scale_factor)


if __name__ == "__main__":
    main()
