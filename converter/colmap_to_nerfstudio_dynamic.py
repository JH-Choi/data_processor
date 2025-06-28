# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/process_data/colmap_converter_to_nerfstudio_dataset.py
# from absl import logging
from typing import Any, Dict, Literal, Optional, Union
# import pycolmap
# from pycolmap import Quaternion

import concurrent.futures
import numpy as np
import cv2
import os
import imageio
import json
from PIL import Image
from scipy import linalg
from pathlib import Path
import torch
import bisect

import sys 
sys.path.append('./')

# from utils.read_write_model import read_cameras_binary, read_images_binary, read_cameras_text, read_images_text

from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_text,
    read_images_text,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_points3D_text,
)
from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils.rich_utils import CONSOLE, status

import pdb

def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    out: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
    }

    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    camera_params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "SIMPLE_RADIAL":
        # f, cx, cy, k

        # r2 = u**2 + v**2;
        # radial = k * r2
        # du = u * radial
        # dv = u * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "RADIAL":
        # f, cx, cy, k1, k2

        # r2 = u**2 + v**2;
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial
        # dv = v * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2

        # uv = u * v;
        # r2 = u**2 + v**2
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial + 2 * p1 * u*v + p2 * (r2 + 2 * u**2)
        # dv = v * radial + 2 * p2 * u*v + p1 * (r2 + 2 * v**2)
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4

        # r = sqrt(u**2 + v**2)

        # if r > eps:
        #    theta = atan(r)
        #    theta2 = theta ** 2
        #    theta4 = theta2 ** 2
        #    theta6 = theta4 * theta2
        #    theta8 = theta4 ** 2
        #    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        #    du = u * thetad / r - u;
        #    dv = v * thetad / r - v;
        # else:
        #    du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

        # u2 = u ** 2
        # uv = u * v
        # v2 = v ** 2
        # r2 = u2 + v2
        # r4 = r2 * r2
        # r6 = r4 * r2
        # radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) /
        #          (1 + k4 * r2 + k5 * r4 + k6 * r6)
        # du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        # dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        # fx, fy, cx, cy, omega
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     thetad = theta * (1 + k * theta2)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     theta4 = theta2 ** 2
        #     thetad = theta * (1 + k * theta2)
        #     thetad = theta * (1 + k1 * theta2 + k2 * theta4)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0
        out["k4"] = 0
        camera_model = CameraModel.OPENCV_FISHEYE
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    out["camera_model"] = camera_model.value
    return out


def create_ply_from_colmap(
    filename: str, recon_dir: Path, output_dir: Path, applied_transform: Union[torch.Tensor, None]
) -> None:
    """Writes a ply file from colmap.

    Args:
        filename: file name for .ply
        recon_dir: Directory to grab colmap points
        output_dir: Directory to output .ply
    """
    if (recon_dir / "points3D.bin").exists():
        colmap_points = read_points3D_binary(recon_dir / "points3D.bin")
    elif (recon_dir / "points3D.txt").exists():
        colmap_points = read_points3D_text(recon_dir / "points3D.txt")
    else:
        raise ValueError(f"Could not find points3D.txt or points3D.bin in {recon_dir}")

    # Load point Positions
    points3D = torch.from_numpy(np.array([p.xyz for p in colmap_points.values()], dtype=np.float32))
    if applied_transform is not None:
        assert applied_transform.shape == (3, 4)
        points3D = torch.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]

    # Load point colours
    points3D_rgb = torch.from_numpy(np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8))

    # write ply
    with open(output_dir / filename, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3D)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")

        for coord, color in zip(points3D, points3D_rgb):
            x, y, z = coord
            r, g, b = color
            f.write(f"{x:8f} {y:8f} {z:8f} {r} {g} {b}\n")


# keep_original_world_coordinate: If True, no extra transform will be applied to world coordinate.
#             Colmap optimized world often have y direction of the first camera pointing towards down direction,
#             while nerfstudio world set z direction to be up direction for viewer.

recon_dir = Path('/mnt/hdd/data/VisDrone/VisDrone-VID/GS_data/uav0000084_00000_v/preprocessed')
colmap_dir = recon_dir / "sparse/0"
ply_filename = str(recon_dir / "sparse/0" / "points3D.ply")
output_dir = recon_dir / "nerfstudio"
keep_original_world_coordinate = False
image_rename_map=None

output_dir.mkdir(parents=True, exist_ok=True)

cam_id_to_camera = read_cameras_text(recon_dir / "sparse/0" / "cameras.txt")
im_id_to_image = read_images_text(recon_dir / "sparse/0" / "images.txt")
if set(cam_id_to_camera.keys()) != {1}:
    CONSOLE.print(f"[bold yellow]Warning: More than one camera is found in {recon_dir}")
    # print(cam_id_to_camera)
    use_single_camera_mode = False  # update bool: one camera per frame
    out = {}  # out = {"camera_model": parse_colmap_camera_params(cam_id_to_camera[1])["camera_model"]}
else:  # one camera for all frames
    out = parse_colmap_camera_params(cam_id_to_camera[1])

frames = []
for im_id, im_data in im_id_to_image.items():
    # NB: COLMAP uses Eigen / scalar-first quaternions
    # * https://colmap.github.io/format.html
    # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
    # the `rotation_matrix()` handles that format for us.

    # TODO(1480) BEGIN use pycolmap API
    # rotation = im_data.rotation_matrix()
    rotation = qvec2rotmat(im_data.qvec)

    translation = im_data.tvec.reshape(3, 1)
    w2c = np.concatenate([rotation, translation], 1)
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
    c2w = np.linalg.inv(w2c)
    # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
    c2w[0:3, 1:3] *= -1
    if not keep_original_world_coordinate:
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1

    name = im_data.name
    if image_rename_map is not None:
        name = image_rename_map[name]
    name = Path(f"./images/{name}")

    frame = {
        "file_path": name.as_posix(),
        "transform_matrix": c2w.tolist(),
        "colmap_im_id": im_id,
    } 

    if not use_single_camera_mode:  # add the camera parameters for this frame
        frame.update(parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id]))

    frames.append(frame)

out["frames"] = frames

applied_transform = None
if not keep_original_world_coordinate:
    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([0, 2, 1]), :]
    applied_transform[2, :] *= -1
    out["applied_transform"] = applied_transform.tolist()

# create ply from colmap
assert ply_filename.endswith(".ply"), f"ply_filename: {ply_filename} does not end with '.ply'"
create_ply_from_colmap(
    ply_filename,
    colmap_dir,
    output_dir,
    torch.from_numpy(applied_transform).float() if applied_transform is not None else None,
)
out["ply_file_path"] = ply_filename

with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=4)








