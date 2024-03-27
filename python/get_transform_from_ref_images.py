import argparse
import pycolmap

import numpy as np



def get_transform_from_ref_images(model, ids_o, ids_x, ids_y):

    # ids_o = 941
    # ids_x = [941, 1034]
    # ids_y = [1939, 1915]

    origin = model.images[ids_o].projection_center()
    x_dir = model.images[ids_x[1]].projection_center() - model.images[ids_x[0]].projection_center()
    y_dir = model.images[ids_y[1]].projection_center() - model.images[ids_y[0]].projection_center()
    z_dir = np.cross(x_dir, y_dir)

    x_dir = x_dir / np.linalg.norm(x_dir)
    y_dir = y_dir / np.linalg.norm(y_dir)
    z_dir = z_dir / np.linalg.norm(z_dir)

    rot = np.stack([x_dir, y_dir, z_dir], axis=1)
    pos = origin

    rot = rot.T
    pos = -np.dot(rot, pos)

    sim = pycolmap.Sim3d(1.0, rot, pos)

    return sim


def main():
    parser = argparse.ArgumentParser(description='Process COLMAP sparse reconstruction')
    parser.add_argument("--input_path", required=True, type=str, help="Path to the input COLMAP model directory")
    parser.add_argument("--out_transform_path", required=True, type=str, help="Path to the output transform file")
    parser.add_argument("--idx_o", required=True, type=int, help="Index of the origin image")
    parser.add_argument("--idx_x", required=True, type=str, help="Indices of the two x-axis image")
    parser.add_argument("--idx_y", required=True, type=str, help="Indices of the two y-axis image")

    args = parser.parse_args()

    input_path = args.input_path
    out_transform_path = args.out_transform_path

    idx_o = args.idx_o
    idx_x = args.idx_x
    idx_y = args.idx_y
    print(idx_x.split(","))
    idx_x = [int(idx.strip()) for idx in args.idx_x.split(",")]
    idx_y = [int(idx.strip()) for idx in args.idx_y.split(",")]


    model = pycolmap.Reconstruction(input_path)
 
    sim = get_transform_from_ref_images(model, idx_o, idx_x, idx_y)

    with open(out_transform_path, 'w') as f:
        for row in sim.matrix():
            f.write(" ".join([str(val) for val in row]) + "\n")
        f.write("0.0, 0.0, 0.0 1.0\n")

if __name__ == '__main__':
    main()



