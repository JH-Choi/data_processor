import argparse
import pycolmap

import os
import subprocess



import time
from tqdm import tqdm


def get_visible_3d_points(image_ids, model):

    start = time.time()
    images = []
    for image_id in image_ids:
        images.append(model.images[image_id])
    
    num_visible_points3D_wd = 0 # with duplication
    for image in images.values():
        num_visible_points3D_wd += image.num_points3D

    visible_point3D_ids_wd = [-1] * num_visible_points3D_wd

    idx = 0
    for image in images.values():
        for point2D in image.points2D:
            if point2D.has_point3D():
                visible_point3D_ids_wd[idx] = point2D.point3D_id
                idx = idx + 1
    assert(idx == num_visible_points3D_wd)
    visible_point3D_ids = set(visible_point3D_ids_wd)

    end = time.time()
    
    print("Processing time for fetchiing visible 3d points : ", end - start, " seconds")
    print("Number of given images: ", len(images))
    print("Number of visible 3D points : ", len(visible_point3D_ids))

    return visible_point3D_ids


def get_covering_images(point3D_ids, model):

    start = time.time()

    points3D = []
    for point3D_id in point3D_ids:
        points3D.append(model.points3D[point3D_id])
    
    num_covering_images_wd = 0 # with duplication
    for point3D in points3D.values():
        num_covering_images_wd += point3D.track.length()

    covering_image_ids_wd = [-1] * num_covering_images_wd
 
    idx = 0
    for point3D in points3D.values():
        for obs in point3D.track.elements:
            covering_image_ids_wd[idx] = obs.image_id
            idx = idx + 1
    assert(idx == num_covering_images_wd)
    covering_image_ids = set(covering_image_ids_wd)

    end = time.time()


    print("Processing time for fetching covering images : ", end - start, " seconds")
    print("Number of given 3D points: ", len(points3D))
    print("Number of covering images : ", len(covering_image_ids))


def main():
    parser = argparse.ArgumentParser(description='Process COLMAP sparse reconstruction')
    parser.add_argument("--input_path", required=True, type=str, help="Path to the input COLMAP model directory")
    parser.add_argument("--output_path", required=True, type=str, help="Path to the split output COLMAP model directory")
    parser.add_argument("--tiles", required=True, type=str, help="Path to the tiles directory")
    parser.add_argument("--init_min_num_observations", default=100, type=int, help="Initial minimum number of observations for filtering")
    parser.add_argument("--desired_max_num_images", default=600, type=int, help="Desired maximum number of images in each split model")
    parser.add_argument("--transform_path", type=str, help="Path to the transform file", default=None)
    parser.add_argument("--bbox_path", type=str, help="Path to the bounding box file", default=None)


    args = parser.parse_args()


    input_path = args.input_path
    output_path = args.output_path

    bounding_box = None
    if args.bbox_path:
        with open(args.bbox_path) as f:
            bounding_box = f.readline().strip()

    transform_path = args.transform_path

    tiles = [int(tile.strip()) for tile in args.tiles.split(",")]
    assert(len(tiles) == 3)

    des_max_num_images = args.desired_max_num_images
    init_min_num_observations = args.init_min_num_observations

    os.makedirs((output_path), exist_ok=True)
    if transform_path is not None:
        subprocess.run(["cp", transform_path, output_path])

    if transform_path is None and bounding_box is None:
        subprocess.call(["colmap", "model_converter", "--input_path", input_path, "--output_path", output_path, "--output_type", "BIN"])
    elif transform_path is not None and bounding_box is None:
        subprocess.run(["colmap", "model_transformer", "--input_path", input_path, "--output_path", output_path, "--transform_path", transform_path])
    elif transform_path is None and bounding_box is not None:
        subprocess.run(["colmap", "model_cropper", "--input_path", input_path, "--output_path", output_path, "--boundary", bounding_box])
    else:
        subprocess.run(["colmap", "model_transformer", "--input_path", input_path, "--output_path", output_path, "--transform_path", transform_path])
        subprocess.run(["colmap", "model_cropper", "--input_path", output_path, "--output_path", output_path, "--boundary", bounding_box])

    input_path = output_path

    subprocess.call(["colmap", "model_splitter", \
            "--split_type", "parts", \
            "--input_path", input_path, \
            "--output_path", output_path, \
            "--min_reg_images", "0", \
            "--min_num_points", "0", \
            "--split_params", f"{tiles[0]},{tiles[1]},{tiles[2]}"])

    num_parts = tiles[0] * tiles[1] * tiles[2]

    ref_model = pycolmap.Reconstruction(input_path)
    ref_image_ids = []
    for image_id in ref_model.images.keys():
        ref_image_ids.append(image_id)


    for i in range(num_parts):

        split_model = pycolmap.Reconstruction(os.path.join(output_path, f"{i}"))
        num_images = split_model.num_images()

        print("")
        print("Processing part ", i)
        print("--------------------")
        trial = 1
        for trial in range(1, 11):
            if(num_images <= des_max_num_images):
                break

            cur_min_num_obs = init_min_num_observations * trial
            subprocess.call(["colmap", "image_filterer", \
                    "--input_path", os.path.join(output_path, f"{i}"), \
                    "--output_path", os.path.join(output_path, f"{i}"), \
                    "--min_num_observations", str(cur_min_num_obs)])
            split_model = pycolmap.Reconstruction(os.path.join(output_path, f"{i}"))
            num_images = split_model.num_images()

        images = split_model.images

        image_ids = []
        for image_id in images.keys():
            image_ids.append(image_id)
        image_ids = sorted(image_ids)

        image_ids_to_remove = []
        for image_id in ref_image_ids:
            if image_id not in image_ids:
                image_ids_to_remove.append(image_id)
        image_ids_to_remove = sorted(image_ids_to_remove)

        # write image_id_list
        with open(os.path.join(output_path, f"{i}", "image_list.txt"), "w") as f:
            for image_id in image_ids:
                f.write(f"{image_id}\n")
        with open(os.path.join(output_path, f"{i}", "image_list_excluded.txt"), "w") as f:
            for image_id in image_ids_to_remove:
                f.write(f"{image_id}\n")

        # remove bbox_oritned.txt and bbox_oriented_exact.txt
        if os.path.exists(os.path.join(output_path, f"{i}", "bbox_oriented.txt")):
            os.remove(os.path.join(output_path, f"{i}", "bbox_oriented.txt"))
        if os.path.exists(os.path.join(output_path, f"{i}", "bbox_oriented_exact.txt")):
            os.remove(os.path.join(output_path, f"{i}", "bbox_oriented_exact.txt"))
        if os.path.exists(os.path.join(output_path, f"{i}", "bbox_aligned_exact.txt")):
            os.remove(os.path.join(output_path, f"{i}", "bbox_aligned_exact.txt"))

        #rename bbox_aligned.txt as bbox_tile.txt
        if os.path.exists(os.path.join(output_path, f"{i}", "bbox_aligned.txt")):
            os.rename(os.path.join(output_path, f"{i}", "bbox_aligned.txt"), os.path.join(output_path, f"{i}", "bbox_tile.txt"))






    for i in tqdm(range(num_parts)):
        subprocess.call(["colmap", "image_deleter", \
                "--input_path", input_path, \
                "--output_path", os.path.join(output_path, f"{i}"), \
                "--image_ids_path", os.path.join(output_path, f"{i}", "image_list_excluded.txt")], stdout=subprocess.DEVNULL)
        os.remove(os.path.join(output_path, f"{i}", "image_list_excluded.txt"))

        split_model = pycolmap.Reconstruction(os.path.join(output_path, f"{i}"))
        
        bbox_scene = split_model.compute_bounding_box()
        with open(os.path.join(output_path, f"{i}", "bbox_scene.txt"), "w") as f:
            f.write(f"{bbox_scene[0][0]}, {bbox_scene[0][1]}, {bbox_scene[0][2]}\n")
            f.write(f"{bbox_scene[1][0]}, {bbox_scene[1][1]}, {bbox_scene[1][2]}\n")


if __name__ == '__main__':
    main()



