# https://github.com/nerfstudio-project/gsplat/blob/main/examples/datasets/colmap.py
import json
import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

colmap_dir = 
if not os.path.exists(colmap_dir):
    colmap_dir = os.path.join(data_dir, "sparse")
assert os.path.exists(
    colmap_dir
), f"COLMAP directory {colmap_dir} does not exist."

manager = SceneManager(colmap_dir)
manager.load_cameras()
manager.load_images()
manager.load_points3D()

# Extract extrinsic matrices in world-to-camera format.
imdata = manager.images
w2c_mats = []
camera_ids = []
Ks_dict = dict()
params_dict = dict()
imsize_dict = dict()  # width, height
mask_dict = dict()
bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
for k in imdata:
    im = imdata[k]
    rot = im.R()
    trans = im.tvec.reshape(3, 1)
    w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
    w2c_mats.append(w2c)

    # support different camera intrinsics
    camera_id = im.camera_id
    camera_ids.append(camera_id)

    # camera intrinsics
    cam = manager.cameras[camera_id]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor
    Ks_dict[camera_id] = K

    # Get distortion parameters.
    type_ = cam.camera_type
    if type_ == 0 or type_ == "SIMPLE_PINHOLE":
        params = np.empty(0, dtype=np.float32)
        camtype = "perspective"
    elif type_ == 1 or type_ == "PINHOLE":
        params = np.empty(0, dtype=np.float32)
        camtype = "perspective"
    if type_ == 2 or type_ == "SIMPLE_RADIAL":
        params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
        camtype = "perspective"
    elif type_ == 3 or type_ == "RADIAL":
        params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
        camtype = "perspective"
    elif type_ == 4 or type_ == "OPENCV":
        params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
        camtype = "perspective"
    elif type_ == 5 or type_ == "OPENCV_FISHEYE":
        params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
        camtype = "fisheye"
    assert (
        camtype == "perspective" or camtype == "fisheye"
    ), f"Only perspective and fisheye cameras are supported, got {type_}"

    params_dict[camera_id] = params
    imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
    mask_dict[camera_id] = None
print(
    f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
)

if len(imdata) == 0:
    raise ValueError("No images found in COLMAP.")
if not (type_ == 0 or type_ == 1):
    print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

w2c_mats = np.stack(w2c_mats, axis=0)

# Convert extrinsics to camera-to-world.
camtoworlds = np.linalg.inv(w2c_mats)

# Image names from COLMAP. No need for permuting the poses according to
# image names anymore.
image_names = [imdata[k].name for k in imdata]

# Previous Nerf results were generated with images sorted by filename,
# ensure metrics are reported on the same test set.
inds = np.argsort(image_names)
image_names = [image_names[i] for i in inds]
camtoworlds = camtoworlds[inds]
camera_ids = [camera_ids[i] for i in inds]