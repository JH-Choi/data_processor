import argparse
import open3d as o3d
import pdb

# Point cloud outlier removal
# https://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html#Statistical-outlier-removal

parser = argparse.ArgumentParser(description='Process COLMAP sparse reconstruction')
parser.add_argument("--pcd_file", type=str, help="PCD file to be processed")
parser.add_argument("--out_file", type=str, help="Output file after SOR")
args = parser.parse_args()

# Prepare input data
pcd_file = args.pcd_file
out_file = args.out_file
print('Processing: ', pcd_file)
print('Output: ', out_file)
pcd = o3d.io.read_point_cloud(pcd_file) # 30990087 points

# voxel_size=0.02
voxel_size=0.0002
print(f"Downsample the point cloud with a voxel of {voxel_size}")
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

print("Statistical oulier removal")
# filter_pcd, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0, print_progress=True)
filter_pcd, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.0, print_progress=True)

# # display_inlier_outlier(voxel_down_pcd, ind)
o3d.io.write_point_cloud(out_file, filter_pcd)
