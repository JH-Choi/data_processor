import argparse
import pycolmap

import numpy as np


def get_transform_from_ref_images(model, ids_o):

    # ids_o = 941

    origin = model.images[ids_o].projection_center()
    print(origin)



    # rot = np.stack([x_dir, y_dir, z_dir], axis=1)
    # 90-degree rotation matrix around the X-axis
    rot = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    pos = origin

    rot = rot.T
    pos = -np.dot(rot, pos)

    print(rot)
    print(pos)

    sim = pycolmap.Sim3d(1.0, rot, pos)

    return sim



def main():
    parser = argparse.ArgumentParser(description='Process COLMAP sparse reconstruction')
    parser.add_argument("--input_path", required=True, type=str, help="Path to the input COLMAP model directory")
    parser.add_argument("--out_transform_path", required=True, type=str, help="Path to the output transform file")
    parser.add_argument("--idx_o", required=True, type=int, help="Index of the origin image")

    args = parser.parse_args()

    input_path = args.input_path
    out_transform_path = args.out_transform_path

    idx_o = args.idx_o


    model = pycolmap.Reconstruction(input_path)
 
    sim = get_transform_from_ref_images(model, idx_o)

    with open(out_transform_path, 'w') as f:
        for row in sim.matrix():
            f.write(" ".join([str(val) for val in row]) + "\n")
        f.write("0.0, 0.0, 0.0 1.0\n")

if __name__ == '__main__':
    main()

