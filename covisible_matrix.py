import os
import numpy as np
import json

from utils.read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary


def compute_reconstruction_statistics(reference_model_path):
    # Load binary files
    cameras = read_cameras_binary(os.path.join(reference_model_path, 'cameras.bin'))
    images = read_images_binary(os.path.join(reference_model_path, 'images.bin'))
    points3D = read_points3D_binary(os.path.join(reference_model_path, 'points3D.bin'))

    # Build images, intrinsics, poses dictionaries
    intrinsics = {}
    poses = {}
    images_dict = {}

    for image_id, image in images.items():
        intrinsics[image.name] = cameras[image.camera_id]  # camera parameters
        poses[image.name] = (image.qvec, image.tvec)       # pose
        images_dict[image.name] = image_id

    # Build covisibility matrix
    max_image_id = max(images.keys())
    n_covisible_points = np.zeros((max_image_id + 1, max_image_id + 1))

    # Map: image_id -> set of visible 3D point ids
    image_visible_points3D = {}
    for image_id, image in images.items():
        point3D_ids = [p for p in image.point3D_ids if p != -1]
        image_visible_points3D[image_id] = set(point3D_ids)

    for id1 in image_visible_points3D:
        for id2 in image_visible_points3D:
            if id1 > id2:
                continue
            common_points = image_visible_points3D[id1] & image_visible_points3D[id2]
            n_covisible_points[id1, id2] = len(common_points)
            n_covisible_points[id2, id1] = n_covisible_points[id1, id2]

    return images_dict, intrinsics, poses, n_covisible_points


if __name__ == "__main__":
    target_views = 4

    # Mipnerf
    # reference_model_path = "/mnt/hdd/data/mipnerf360_data/"
    # split_file = "mipnerf_split_list.json"
    # out_file = "mipnerf_views_dict.json" # last file is the closest to the reference image

    # Tanks and Temples
    reference_model_path = "/mnt/hdd/data/instantsplat_data/eval/Tanks"
    split_file = "tanks_split_list.json"
    out_file = "tanks_views_dict.json"

    target_views_dict = {}

    split_dict = json.load(open(split_file))
    for split_name in split_dict.keys():
        split_model_path = os.path.join(reference_model_path, split_name, 'sparse/0')
        # print(split_model_path)
        images, intrinsics, poses, n_covisible_points = compute_reconstruction_statistics(split_model_path)

        ref_img_candidates = split_dict[split_name]
        ref_img_indexs = [images[img_name] for img_name in ref_img_candidates]

        target_views_dict[split_name] = {}

        for e_i, ref_img_idx in enumerate(ref_img_indexs):
            ref_img_covisible_matrix = n_covisible_points[ref_img_idx, :]

            # choose the top 4 images with the most covisible points
            # Exclude the reference image itself
            top_images = np.argsort(ref_img_covisible_matrix)[-target_views-2:-1]

            images_names, images_values = list(images.keys()), list(images.values())
            top_images_names = [images_names[images_values.index(img_idx)] for img_idx in top_images]
            target_views_dict[split_name][ref_img_candidates[e_i]] = top_images_names


    # save target views dict
    with open(out_file, "w") as f:
        json.dump(target_views_dict, f, indent=4)


