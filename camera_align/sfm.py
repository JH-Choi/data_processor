import os
import logging

# conda activate colmap_calibration

# source_path = '/mnt/hdd/code/Dongki_project/Genesis3DGS/data/IMG_0687/collect_kf2_sth50/'
# source_path = '/mnt/hdd/code/calib_practice/colmap_handeye/example_data'
# source_path = '/mnt/hdd/code/gaussian_splatting/data_processor/iphone_extract/IMG_0701'
source_path = '/mnt/hdd/code/Dongki_project/Genesis3DGS/data/IMG_0701'

# pycolmap = 0.4.0
from aruco_estimator.aruco_scale_factor import ArucoScaleFactor
from colmap_wrapper.colmap import COLMAP
aruco_size = 0.0995 # the size of the aruco marker in meters

# Load Colmap project folder
project = COLMAP(project_path=source_path)

# Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
aruco_scale_factor = ArucoScaleFactor(photogrammetry_software=project, aruco_size=aruco_size)
aruco_distance, aruco_corners_3d = aruco_scale_factor.run()
print('Size of the unscaled aruco markers: ', aruco_distance)

# Calculate scaling factor, apply it to the scene and save scaled point cloud
dense, scale_factor = aruco_scale_factor.apply() 
print('Point cloud and poses are scaled by: ', scale_factor)
print('Size of the scaled (true to scale) aruco markers in meters: ', aruco_distance * scale_factor)

# Write Data
aruco_scale_factor.write_data()