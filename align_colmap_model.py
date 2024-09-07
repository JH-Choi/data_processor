import argparse
import pycolmap

import os
import subprocess






def main():
    parser = argparse.ArgumentParser(description='Process COLMAP sparse reconstruction')
    parser.add_argument("--input_path", required=True, type=str, help="Path to the input COLMAP model directory")
    parser.add_argument("--output_path", required=True, type=str, help="Path to the output COLMAP model directory")
    parser.add_argument("--ref_path", required=True, type=str, help="Path to the reference COLMAP model directory")
    parser.add_argument("--robust_alignment_max_error", type=float, default=10.0, help="Robust alignment max error")

    args = parser.parse_args()


    input_path = args.input_path
    output_path = args.output_path
    ref_path = args.ref_path

    os.makedirs((output_path), exist_ok=True)


    ref_model = pycolmap.Reconstruction(ref_path)
    input_model = pycolmap.Reconstruction(input_path)

    ref_names = [image.name for image in ref_model.images.values()]
    input_names = [image.name for image in input_model.images.values()]

    shared_names = list(set(ref_names) & set(input_names))
    shared_images = [ref_model.find_image_with_name(name) for name in shared_names]

    # write the list to a text file [name] [pos x] [pos y] [pos z]
    with open(os.path.join(output_path, 'ref_image_list.txt'), 'w') as f:
        for image in shared_images:
            name = image.name
            pos = image.projection_center()

            f.write(f"{name} {pos[0]} {pos[1]} {pos[2]}\n")

    # colmap model_aligner --input_path sfm --output_path test --ref_images_path test/ref_image_list.txt --ref_is_gps 0 --transform_path test/transform.txt --robust_alignment_max_error 10.0

    subprocess.run(['colmap', 'model_aligner', '--input_path', input_path, \
            '--output_path', output_path, \
            '--ref_images_path', os.path.join(output_path, 'ref_image_list.txt'), \
            '--ref_is_gps', '0', '--transform_path', os.path.join(output_path, 'transform.txt'), \
            '--robust_alignment_max_error', str(args.robust_alignment_max_error)])

if __name__ == '__main__':
    main()



