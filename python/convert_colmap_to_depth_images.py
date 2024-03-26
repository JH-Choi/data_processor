import numpy as np
import open3d as o3d
import argparse
import cv2

from typing import Dict


from multiprocessing import Pool

import os



colmap_images : Dict[int, Image]
colmap_points : Dict[int, Point3D]
colmap_cameras : Dict[int, Camera]


output_depth_dir = None

def process_image(image_id):
    image = colmap_images[image_id]

    sparse_depth_rgb_data = np.array(image.keypoints, dtype=float)


    if os.path.exists(os.path.join(output_depth_dir, f"{image.image_id:06d}.npy")):
        print(f"Skipping {image.image_id}")
        return
    np.save(os.path.join(output_depth_dir, f"{image.image_id:06d}.npy"), sparse_depth_rgb_data)

def main():
    parser = argparse.ArgumentParser(description="Visualize a depth image as a point cloud using Open3D.")
    parser.add_argument("--model_path", required=True, help="Path to the colmap model directory")
    parser.add_argument("--output_depth_dir", required=True, help="Path to the directory where the depth images are stored")
    args = parser.parse_args()

    os.makedirs(args.output_depth_dir, exist_ok=True)

    global colmap_images, colmap_points, colmap_cameras
    global output_depth_dir

    output_depth_dir = args.output_depth_dir

    print("Reading colmap model")
    colmap_images, colmap_points, colmap_cameras = read_colmap_model(os.path.join(args.model_path, "images.txt"), os.path.join(args.model_path, "points3D.txt"), os.path.join(args.model_path, "cameras.txt"))
    print("Finished reading colmap model")

    poses = {}
    for image in colmap_images.values():        
        poses[image.image_id] = image.T_wc

    with open(os.path.join(args.output_depth_dir, "poses.txt"), "w") as file:
        file.write("### id T00 T01 T02 T03 T10 T11 T12 T13 T20 T21 T22 T23 T30 T31 T32 T33\n")
        for image_id, pose in poses.items():
            file.write(f"{image_id} ")
            for row in pose:
                for val in row:
                    file.write(f"{val} ")
            file.write("\n")

    with Pool(processes=16) as pool:
        pool.map(process_image, colmap_images.keys())


    #     camera = colmap_cameras[image.camera_id]
    #     depth_image = np.zeros((camera.height, camera.width), dtype=np.float32)
    #     color_image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)

    #     for keypoint in image.keypoints:
    #         point3d_id = keypoint[2]
    #         u = int(keypoint[0])
    #         v = int(keypoint[1])
    #         point3d = colmap_points[point3d_id]

    #         pts_w = np.array([point3d.x, point3d.y, point3d.z])
    #         pts_c = R_cw @ pts_w + t_cw
    #         depth = pts_c[2]
    #         depth_image[v,u] = depth

    #         color_image[v,u] = np.array([point3d.r, point3d.g, point3d.b])

    #     np.save(os.path.join(args.output_depth_dir, f"{image.image_id}.npy"), depth_image)
    #     cv2.imwrite(os.path.join(args.output_depth_dir, f"{image.image_id}.png"), color_image)

        # # Convert the depth image to an Open3D depth image
        # o3d_depth_image = o3d.geometry.Image(depth_image.astype(np.float32))
        # o3d_color_image = o3d.geometry.Image(color_image.astype(np.uint8))

        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     o3d_color_image, o3d_depth_image, depth_scale=1.0, depth_trunc=9999999.0, convert_rgb_to_intensity=False)

        # intrinsics = o3d.camera.PinholeCameraIntrinsic(width=camera.width, height=camera.height,
        #                                                fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy)

        # 
        # # Create point cloud from RGBD image
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        # # pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth_image, intrinsics)

        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
