import os
import sys
import collections
import numpy as np
import argparse
from utils.read_write_model import read_model, write_model
import pdb


def main():
    parser = argparse.ArgumentParser(description="Read and write COLMAP binary and text models")
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--output_model",
                        help="path to output model folder")
    parser.add_argument("--output_format", choices=[".bin", ".txt"],
                        help="outut model format", default=".txt")
    args = parser.parse_args()


    # args.input_model = '/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/sparse/0'

    cameras, images, points3D = read_model(path=args.input_model, ext=args.input_format)

    print("[Before] num_cameras:", len(cameras))
    print("[Before] num_images:", len(images))
    print("[Before] num_points3D:", len(points3D))
    new_images = {}
    for key, val in images.items():
        if args.split not in val.name:
            continue
        else:
            new_images[key] = val

    print("[After] num_cameras:", len(cameras))
    print("[After] num_images:", len(new_images))
    print("[After] num_points3D:", len(points3D))

    if args.output_model is not None:
        write_model(cameras, new_images, points3D, path=args.output_model, ext=args.output_format)

if __name__ == "__main__":
    main()