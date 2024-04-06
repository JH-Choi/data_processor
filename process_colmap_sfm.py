import argparse

import subprocess
import os

import pycolmap


def main():
    parser = argparse.ArgumentParser(description='Process COLMAP sparse reconstruction')
    parser.add_argument("--colmap_dir", required=True, type=str, help="Path to the colmap directory")
    parser.add_argument("--image_path", required=True, type=str, help="Path to the dataset directory")
    args = parser.parse_args()

    image_path = args.image_path
    colmap_dir = args.colmap_dir

    model = pycolmap.Reconstruction(colmap_dir)

    image_list = os.path.join(colmap_dir, "image_list.txt")
    database_path = os.path.join(colmap_dir, "database.db")

    #first value of camera dict

    cameras = []
    for camera in model.cameras.values():
        cameras.append(camera)

    assert(len(cameras) == 1)
    camera = cameras[0]

#     assert(pycolmap.CameraModelId("PINHOLE") == camera.model)

    fx = camera.focal_length_x
    fy = camera.focal_length_y
    cx = camera.principal_point_x
    cy = camera.principal_point_y

    subprocess.call(["colmap", "feature_extractor", \
            "--image_path", image_path, \
            "--image_list", image_list, \
            "--database_path", database_path, \
            "--SiftExtraction.use_gpu", "0", \
            "--ImageReader.single_camera", "1", \
            "--ImageReader.camera_model", "PINHOLE", \
            f"--ImageReader.camera_params={fx},{fy},{cx},{cy}"])

    subprocess.call(["colmap", "spatial_matcher", \
            "--database_path", database_path, \
            "--SiftMatching.min_num_inliers", "50", \
            "--SpatialMatching.is_gps", "0", \
            "--SpatialMatching.max_distance", "100"])

    subprocess.call(["colmap", "point_triangulator", \
            "--database_path", database_path, \
            "--image_path", image_path, \
            "--input_path", colmap_dir, \
            "--output_path", colmap_dir, \
            "--Mapper.ba_refine_focal_length", "1", \
            "--Mapper.ba_refine_principal_point", "0", \
            "--Mapper.ba_refine_extra_params", "0"])

if __name__ == '__main__':
    main()


