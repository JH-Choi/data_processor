import argparse
import pycolmap

import os
import subprocess



import time
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser(description='Process COLMAP sparse reconstruction')
    parser.add_argument("--input_path", required=True, type=str, help="Path to the input COLMAP model directory")
    parser.add_argument("--output_path", required=True, type=str, help="Path to the split output COLMAP model directory")
    parser.add_argument("--transform_path", type=str, help="Path to the transform file", default=None)
    args = parser.parse_args()


    input_path = args.input_path
    output_path = args.output_path
    transform_path = args.transform_path

    assert os.path.exists(transform_path) and os.path.exists(input_path)

    os.makedirs((output_path), exist_ok=True)

    subprocess.run(["colmap", "model_transformer", "--input_path", input_path, "--output_path", output_path, "--transform_path", transform_path])


if __name__ == '__main__':
    main()



