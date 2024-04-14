import argparse
import open3d as o3d
import numpy as np
import os
import pdb

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--input_file", required=True, help="Input point cloud or mesh file")
    Parser.add_argument("--output_file", required=True, help="Output point cloud or mesh file")
    Parser.add_argument("--transform_path", help="Path to the transform file to be applied to inputs", default=None)
    Parser.add_argument("--Dtype", help="3D Model type, mesh or pcd", default=None, choices=["mesh", "pcd"])
    args = Parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    transform_path = args.transform_path
    Dtype = args.Dtype
    print("Input file : ", input_file)
    print("Output file : ", output_file)
    print("Transform file : ", transform_path)

    assert os.path.exists(transform_path) and os.path.exists(input_file)

    # read transform
    trasnform = np.loadtxt(transform_path, delimiter=" ")

    if Dtype == "mesh":
        mesh = o3d.io.read_triangle_mesh(input_file)
        mesh.transform(trasnform)
        o3d.io.write_triangle_mesh(output_file, mesh)
    elif Dtype == "pcd":
        pcd = o3d.io.read_point_cloud(input_file)
        pcd.transform(trasnform)
        o3d.io.write_point_cloud(output_file, pcd)

if __name__ == "__main__":
    main()

