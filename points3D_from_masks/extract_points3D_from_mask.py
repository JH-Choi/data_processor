import numpy as np
import cv2
import os
import sys
import open3d as o3d
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from utils.read_write_model import read_model  # from colmap_read_model

root_path = '/mnt/hdd/code/Dongki_project/Genesis3DGS/data/IMG_0687/collect_kf2_sth50/'
colmap_path = os.path.join(root_path, 'sparse', '0')
mask_dir = os.path.join(root_path, 'masks')

# Load COLMAP model
cameras, images, points3D = read_model(colmap_path, ext=".bin")  # or ".txt"

# image_masks = {}
# for image_id, image in images.items():
#     mask_path = os.path.join(mask_dir, image.name[:-4] + ".npz")  # adjust if naming differs
#     mask = np.load(mask_path)['arr_0'].astype(np.int16)
#     if mask is None:
#         raise FileNotFoundError(f"Missing mask for image: {image.name}")
#     image_masks[image_id] = mask > 0  # binary mask

# Collect filtered 3D points
selected_point_ids = set()

for image in images.values():
    image_name = image.name
    print(image_name)
    # mask_path = os.path.join(mask_dir, image_name + ".png")
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0

    mask_path = os.path.join(mask_dir, image.name[:-4] + ".npz")  # adjust if naming differs
    mask = np.load(mask_path)['arr_0'].astype(np.int16)[0,0]
    if mask is None:
        raise FileNotFoundError(f"Missing mask for image: {image.name}")
    # image_masks[image_id] = mask > 0  # binary mask
    mask = mask > 0


    for i, (xy, pid) in enumerate(zip(image.xys, image.point3D_ids)):
        if pid == -1:
            continue  # No 3D point

        x, y = int(round(xy[0])), int(round(xy[1]))
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            if mask[y, x]:
                selected_point_ids.add(pid)

# Get 3D coordinates
selected_points = np.array([points3D[pid].xyz for pid in selected_point_ids])

pcd = o3d.geometry.PointCloud()
import pdb; pdb.set_trace()
pcd.points = o3d.utility.Vector3dVector(np.array(selected_points))
o3d.io.write_point_cloud("masked_points.ply", pcd)

# # Load mask image (same size as original image)
# mask = cv2.imread("path/to/mask.png", cv2.IMREAD_GRAYSCALE) > 0  # Boolean mask

# # Choose your image by name
# target_image_name = "image_name.jpg"

# Get image ID and object
# for cam in cameras.values():
#     print(cam.name)

# for img in images.values():
#     print(img)
#     if img.name == target_image_name:
#         image = img
#         break
# if image is None:
#     raise ValueError("Image not found.")

# # Collect matching 3D point IDs
# selected_point_ids = []

# for point3D_id, point in points3D.items():
#     print(point.track)
    # for track in point.track.elements:
    #     print(track)
        # if track.image_id == image.image_id:
        #     x, y = image.xys[track.point2D_idx]
        #     if mask[int(round(y)), int(round(x))]:  # Check if in mask
        #         selected_point_ids.append(point3D_id)
        #     break  # Only need to check once for this image

# # Now selected_point_ids contains the 3D points inside the mask

# # Optionally extract the 3D point coordinates
# selected_points = [points3D[pid].xyz for pid in selected_point_ids]
# selected_points = np.array(selected_points)