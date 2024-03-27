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

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    tiles = [int(tile.strip()) for tile in args.tiles.split(",")]
    des_max_num_images = args.desired_max_num_images
    init_min_num_observations = args.init_min_num_observations

    assert(len(tiles) == 3)

    os.makedirs((output_path), exist_ok=True)

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

        # colmap image_filterer --input_path . --output_path . --min_num_observations 1000

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

        image_ids_to_remove = []
        for image_id in ref_image_ids:
            if image_id not in image_ids:
                image_ids_to_remove.append(image_id)

        # write image_id_list
        with open(os.path.join(output_path, f"{i}", "image_list.txt"), "w") as f:
            for image_id in image_ids:
                f.write(f"{image_id}\n")
        with open(os.path.join(output_path, f"{i}", "image_list_excluded.txt"), "w") as f:
            for image_id in image_ids_to_remove:
                f.write(f"{image_id}\n")


    for i in tqdm(range(num_parts)):
        subprocess.call(["colmap", "image_deleter", \
                "--input_path", input_path, \
                "--output_path", os.path.join(output_path, f"{i}"), \
                "--image_ids_path", os.path.join(output_path, f"{i}", "image_list_excluded.txt")], stdout=subprocess.DEVNULL)

        

    # get_visible_3d_points(model.images, model)
    # get_covering_images(model.points3D, model)



if __name__ == '__main__':
    main()



