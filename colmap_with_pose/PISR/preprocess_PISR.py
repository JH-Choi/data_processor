import numpy as np
import argparse
import cv2
import os
import sys
import pdb 
from scipy.spatial.transform import Rotation as R
from database import COLMAPDatabase


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


def convert_to_opencv(pose):
    # Coordinate transformation matrix from NeuS to OpenCV
    # Assuming input pose is cam-to-world (as in NeuS), and we want world-to-cam (OpenCV)
    # NeuS: X(right), Y(down), Z(front) --> OpenCV: X(right), Y(down), Z(forward)
    # Flip Z and Y to get to OpenCV convention (left-handed)
    c = np.eye(4)
    c[1,1] = -1  # Flip Y
    c[2,2] = -1  # Flip Z

    pose_opencv = c @ pose  # apply flip
    return pose_opencv

# def create_cameras_txt(intrinsic_file, output_folder, width, height):
#     """Convert intrinsics.txt to COLMAP cameras.txt format"""
#     # Read intrinsics
#     with open(intrinsic_file, 'r') as f:
#         intrinsics = f.readline().strip().split(',')
#         intrinsics = [float(x) for x in intrinsics]
    
#     # Extract parameters
#     fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics
    
#     # Create cameras.txt
#     cameras_file = os.path.join(output_folder, "cameras.txt")
#     with open(cameras_file, "w") as f:
#         f.write("# Camera list with one line of data per camera:\n")
#         f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
#         # Using SIMPLE_RADIAL model which includes: fx, cx, cy, k1
#         f.write(f"1 SIMPLE_RADIAL {width} {height} {fx} {cx} {cy} {k1}\n")

def load_K_Rt_from_P(P):
    """
    modified from IDR https://github.com/lioryariv/idr
    """
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]          # world see cam
    t = out[2]          # cam see world

    K = K/K[2,2]

    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose     #  cam see world

parser = argparse.ArgumentParser() 
parser.add_argument("--data_dir", required=True, help="ScanNet sens unpack dir") 
parser.add_argument("--output_folder", required=True, help="Output nerf-synthetic style data")
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)
image_folder = os.path.join(args.data_dir, "image")
in_cameras_file = os.path.join(args.data_dir, "cameras_sphere.npz")
intrinsic_file = os.path.join(args.data_dir, "intrinsics.txt")
db_file = os.path.join(args.data_dir, "database.db")
debug_file = os.path.join(args.data_dir, "colmap_output.txt")

# read all images
images = os.listdir(image_folder)
images = [os.path.join(image_folder, image) for image in images]
image = cv2.imread(images[0])
width = image.shape[1]
height = image.shape[0]


# Create cameras.txt in COLMAP format
with open(intrinsic_file, 'r') as f:
    intrinsics = f.readline().strip().split(',')
    intrinsics = [float(x) for x in intrinsics]

# Extract parameters
fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics

# Load cameras_sphere.npz
normalized_cameras = np.load(in_cameras_file)
pinhole_dict = {}
n_imgs = int(len(normalized_cameras) / 3)
# scale_mat_0, world_mat_0, pose_mat_0
# how to print all keys of normalized_cameras
for i in range(0, n_imgs):
    # filename is 000, 0001 ... using i 
    image_name = "%03d.png" % i

    world_mat = normalized_cameras['world_mat_%d' % i].astype(np.float32)
    scale_mat = normalized_cameras['scale_mat_%d' % i].astype(np.float32)

    P = world_mat @ scale_mat       # 여기서 normalize 하는 건가. projection matrix 에 미리 scale 을 넣어놔서 반대로 camera 가 unit sphere 안으로 들어오는 효과??
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(P)      # pose: cam see world (like colmap format)
    # NeuS DTU data coordinate system (right down front) is different from blender
    # can we convert to opencv coordinate system (left up front)
    pose = convert_to_opencv(pose)
    pose = np.linalg.inv(pose)  # world see cam

    tx = pose[0, 3]
    ty = pose[1, 3]
    tz = pose[2, 3]
    r = R.from_matrix(pose[:3, :3])
    qx, qy, qz, qw = r.as_quat()
    quat = np.array([qw, qx, qy, qz], dtype=np.float32)
    t = np.array([tx, ty, tz], dtype=np.float32)

    pinhole_dict[image_name] = {}
    pinhole_dict[image_name]["width"] = width
    pinhole_dict[image_name]["height"] = height
    pinhole_dict[image_name]["fx"] = fx
    pinhole_dict[image_name]["fy"] = fy
    pinhole_dict[image_name]["cx"] = cx
    pinhole_dict[image_name]["cy"] = cy
    pinhole_dict[image_name]["k1"] = k1
    pinhole_dict[image_name]["quaternions"] = quat
    pinhole_dict[image_name]["translations"] = t


run_sift_matching(image_folder, db_file, debug_file)
create_init_files(pinhole_dict, db_file, args.output_folder)
run_point_triangulation(image_folder, db_file, args.output_folder, debug_file)
# os.system(f'{colmap_bin} image_undistorter --image_path {image_dir} --input_path {out_dir} --output_path {out_dir} > {debug_file}')
 
