import argparse
import cv2
import os
import os.path as osp
import subprocess
import shutil
from database import COLMAPDatabase
from PIL import Image
from scipy.spatial.transform import Rotation
import numpy as np
import pdb

gpu_index = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
colmap_bin = '/usr/local/bin/colmap'


def run_sift_matching(img_dir, db_file, debug_file=None):
    print('Running sift matching...')

    # feature extraction
    # if there's no attached display, cannot use feature extractor with GPU
    cmd = f'{colmap_bin} feature_extractor --database_path {db_file} \
                                    --image_path {img_dir} \
                                    --ImageReader.camera_model PINHOLE \
                                    --SiftExtraction.max_image_size 5000  \
                                    --SiftExtraction.estimate_affine_shape 0 \
                                    --SiftExtraction.domain_size_pooling 1 \
                                    --SiftExtraction.num_threads 32 \
                                    --SiftExtraction.use_gpu 0 \
                                    --SiftExtraction.gpu_index {gpu_index} > {debug_file}'
    os.system(cmd)

    # feature matching
    cmd = f'{colmap_bin} exhaustive_matcher --database_path {db_file} \
                                     --SiftMatching.guided_matching 1 \
                                     --SiftMatching.use_gpu 0 \
                                     --SiftMatching.gpu_index {gpu_index} > {debug_file}'
    os.system(cmd)


def run_point_triangulation(img_dir, db_file, out_dir, debug_file=None):
    print('Running point triangulation...')

    # triangulate points
    cmd = f'{colmap_bin} point_triangulator  --database_path {db_file} \
                                      --image_path {img_dir} \
                                      --input_path {out_dir} \
                                      --output_path {out_dir} \
                                      --Mapper.tri_ignore_two_view_tracks 1 > {debug_file}'
    os.system(cmd)


def create_init_files(pinhole_dict, db_file, out_dir):

    template = {}
    cameras_line_template = '{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params["width"]
        h = params["height"]
        fx = params["fx"]
        fy = params["fy"]
        cx = params["cx"]
        cy = params["cy"]
        qvec = params["quaternions"]
        tvec = params["translations"]

        img_name = os.path.basename(img_name)

        cam_line = cameras_line_template.format(camera_id="{camera_id}", width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                            tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}", image_name=img_name)
        template[img_name] = (cam_line, img_line)

    # read database
    db = COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    cameras_txt_lines = []
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        print(img_name)
        print(template)
        camera_line = template[img_name][0].format(camera_id=img_id)
        cameras_txt_lines.append(camera_line)

        image_line = template[img_name][1].format(image_id=img_id, camera_id=img_id)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()


def main(args):
    print('Start processing...')
    print('data_dir: {}'.format(args.data_dir))

    num_views = 10 

    # scene_names = os.listdir(args.data_dir)
    # scene_names = sorted(scene_names)
    scene_names = ['0']
    for scene_name in scene_names:
        scene_dir = osp.join(args.data_dir, scene_name)
        print('scene_dir: {}'.format(scene_dir))

        out_dir = osp.join(args.out_dir, scene_name, 'sparse', '0')
        os.makedirs(out_dir, exist_ok=True)
        out_undistorted_dir = os.path.join(args.out_dir, scene_name, 'undistorted')
        os.makedirs(out_undistorted_dir, exist_ok=True)

        debug_file = os.path.join(args.out_dir, scene_name, 'colmap_output.txt')
        db_file = os.path.join(args.out_dir, scene_name, 'database.db')
        image_dir = os.path.join(args.out_dir, scene_name, 'images')
        os.makedirs(image_dir, exist_ok=True)
 
        ref_img_path = osp.join(scene_dir, 'ref_00.png')
        ref_depth_path = osp.join(scene_dir, 'ref_00_depth.png')
        ref_intr_path = osp.join(scene_dir, "ref_00_intrinsic.txt")
        ref_extr_path = osp.join(scene_dir, "ref_00_pose.txt")
        ref_intr = np.loadtxt(ref_intr_path, dtype=np.float32)
        ref_extr = np.loadtxt(ref_extr_path, dtype=np.float32)

        image_names = []
        image_names.append(ref_img_path)
        for i in range(1, num_views + 1):
            src_img_path = osp.join(scene_dir, f"src_{i:02d}.png")
            image_names.append(src_img_path)
        
        for image_path in image_names:
            image_name = os.path.basename(image_path)
            shutil.copy(image_path, os.path.join(image_dir, image_name))
 
        ref_depth = cv2.imread(ref_depth_path, -1)
        depth_height, depth_width = ref_depth.shape
        ref_img = cv2.imread(ref_img_path)
        img_height, img_width = ref_img.shape[:2]
        print("ref_depth shape and ref_img shape", ref_depth.shape, ref_img.shape)
        scale_width, scale_height = img_width / depth_width, img_height / depth_height
        fx, fy = ref_intr[0, 0] * scale_width, ref_intr[1, 1] * scale_height
        cx, cy = ref_intr[0, 2] * scale_width, ref_intr[1, 2] * scale_height
        R, t = ref_extr[:3, :3], ref_extr[:3, 3]
        # convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # scipy returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

       
        pinhole_dict = {}
        pinhole_dict[ref_img_path] = {}
        pinhole_dict[ref_img_path]["width"] = img_width
        pinhole_dict[ref_img_path]["height"] = img_height
        pinhole_dict[ref_img_path]["fx"] = fx
        pinhole_dict[ref_img_path]["fy"] = fy
        pinhole_dict[ref_img_path]["cx"] = cx
        pinhole_dict[ref_img_path]["cy"] = cy
        pinhole_dict[ref_img_path]["quaternions"] = quat
        pinhole_dict[ref_img_path]["translations"] = t


        for i in range(1, num_views + 1):
            src_img_path = osp.join(scene_dir, f"src_{i:02d}.png")
            src_intr_path = osp.join(scene_dir, f"src_{i:02d}_intrinsic.txt")
            src_extr_path = osp.join(scene_dir, f"src_{i:02d}_pose.txt")
            src_intr = np.loadtxt(src_intr_path, dtype=np.float32)
            src_extr = np.loadtxt(src_extr_path, dtype=np.float32)
            fx, fy = src_intr[0, 0] * scale_width, src_intr[1, 1] * scale_height
            cx, cy = src_intr[0, 2] * scale_width, src_intr[1, 2] * scale_height
            R, t = src_extr[:3, :3], src_extr[:3, 3]
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat()  # scipy returns [x, y, z, w]
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

            pinhole_dict[src_img_path] = {}
            pinhole_dict[src_img_path]["width"] = img_width
            pinhole_dict[src_img_path]["height"] = img_height
            pinhole_dict[src_img_path]["fx"] = fx
            pinhole_dict[src_img_path]["fy"] = fy
            pinhole_dict[src_img_path]["cx"] = cx
            pinhole_dict[src_img_path]["cy"] = cy
            pinhole_dict[src_img_path]["quaternions"] = quat
            pinhole_dict[src_img_path]["translations"] = t

        run_sift_matching(image_dir, db_file, debug_file)
        create_init_files(pinhole_dict, db_file, out_dir)
        # run_point_triangulation(image_dir, db_file, out_dir, debug_file)
        # os.system(f'{colmap_bin} image_undistorter --image_path {image_dir} --input_path {out_dir} --output_path {out_dir} > {debug_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()
    # args.data_dir = '/mnt/hdd/data/MoRe_data/MVD/extracted/eth3d/'
    # args.out_dir = '/mnt/hdd/data/MoRe_data/MVD/GS_data/eth3d/'
    args.data_dir = '/mnt/hdd/data/MoRe_data/MVD/extracted/dtu/'
    args.out_dir = '/mnt/hdd/data/MoRe_data/MVD/GS_data/dtu/'
    main(args)