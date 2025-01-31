import argparse

import subprocess
import os


def main():
    parser = argparse.ArgumentParser(description='Process COLMAP sparse reconstruction')
    parser.add_argument("--colmap_dir", required=True, type=str, help="Path to the colmap directory")
    parser.add_argument("--image_path", required=True, type=str, help="Path to the dataset directory")
    args = parser.parse_args()

    image_path = args.image_path
    colmap_dir = args.colmap_dir
    gpu_idx = 0

    sparse_dir = os.path.join(colmap_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    database_path = os.path.join(colmap_dir, "database.db")

    # feature extraction
    subprocess.call(["colmap", "feature_extractor", \
        "--image_path", image_path, \
        "--database_path", database_path, \
        "--SiftExtraction.use_gpu", "1", \
        "--SiftExtraction.gpu_index", f"{gpu_idx}", \
        "--ImageReader.single_camera", "1", \
        "--ImageReader.camera_model", "SIMPLE_RADIAL" ])

    # feature matching
    subprocess.call(["colmap", "exhaustive_matcher", \
        "--database_path", database_path, \
        "--SiftMatching.use_gpu", "1", \
        "--SiftMatching.guided_matching", "1", \
        "--SiftMatching.max_num_matches", "65536", \
        "--SiftMatching.max_error", "3", \
        "--SiftMatching.gpu_index", f"{gpu_idx}" \
        ])

    # Running SfM
    subprocess.call(["colmap", "mapper", \
        "--database_path", database_path, \
        "--image_path", image_path, \
        "--output_path", sparse_dir, \
        "--Mapper.tri_min_angle", "1.5", \
        "--Mapper.filter_min_tri_angle", "1.5", \
        "--Mapper.ba_global_max_num_iterations", "50", \
        "--Mapper.ba_global_max_refinements", "5"
        ])
        # "--Mapper.ba_local_max_num_iterations", "15" \
    # if dataset includes large number of images, use the following options to speed up the process
    # tri_min_angle: 3.0 / filter_min_tri_angle 3.0 / ba_global_max_num_iterations: 25 / ba_global_max_refinements: 3
    # Default options
    # tri_min_angle: 1.5 / filter_min_tri_angle 1.5 / ba_global_max_num_iterations: 50 / ba_global_max_refinements: 5
    

if __name__ == '__main__':
    main()


