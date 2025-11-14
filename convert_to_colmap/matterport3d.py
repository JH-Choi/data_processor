import os 
import cv2
import numpy as np 
from pathlib import Path
from tqdm import tqdm
from glob import glob
from scipy.spatial.transform import Rotation as R
import sqlite3
import struct
import imageio
import trimesh


def create_tables(conn):
    """Create tables in the database."""
    sql_create_images_table = """CREATE TABLE IF NOT EXISTS images (
                                    image_id INTEGER PRIMARY KEY,
                                    name TEXT NOT NULL,
                                    camera_id INTEGER,
                                    prior_qw REAL,
                                    prior_qx REAL,
                                    prior_qy REAL,
                                    prior_qz REAL,
                                    prior_tx REAL,
                                    prior_ty REAL,
                                    prior_tz REAL,
                                    FOREIGN KEY (camera_id) REFERENCES cameras (camera_id)
                                );"""
    sql_create_cameras_table = """CREATE TABLE IF NOT EXISTS cameras (
                                    camera_id INTEGER PRIMARY KEY,
                                    model INTEGER NOT NULL,
                                    width INTEGER NOT NULL,
                                    height INTEGER NOT NULL,
                                    params BLOB NOT NULL,
                                    prior_focal_length INTEGER
                                );"""
    try:
        cursor = conn.cursor()
        cursor.execute(sql_create_images_table)
        cursor.execute(sql_create_cameras_table)
    except Exception as e:
        print(e)



def create_connection(db_file):
    """Create a database connection to the SQLite database specified by db_file."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)
    return conn


def insert_camera(conn, camera_id, model, width, height, params, prior_focal_length):
    """Insert a new camera into the cameras table."""
    # Pack parameters into a binary blob
    params_blob = struct.pack(f'{"d" * len(params)}', *params)
        
    sql = '''INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
             VALUES(?,?,?,?,?,?)'''
    cursor = conn.cursor()
    cursor.execute(sql, (camera_id, model, width, height, params_blob, prior_focal_length))


def insert_image(conn, image_data):
    """Insert a new image into the images table."""
    sql = '''INSERT INTO images(image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz)
             VALUES(?,?,?,?,?,?,?,?,?,?)'''
    cursor = conn.cursor()
    cursor.execute(sql, image_data)



root_folder = '/mnt/hdd/data/matterport/v1/scans'
scene_name = '2t7WUuJeko7'

scene_path = os.path.join(root_folder, scene_name)
images_path = os.path.join(scene_path, 'matterport_skybox_images')
depths_path = os.path.join(scene_path, 'dk_skybox_depth_images')
poses_path = os.path.join(scene_path, 'dk_skybox_camera_poses')
ply_path = os.path.join(scene_path, f'{scene_name}_10.ply')

images = glob(os.path.join(images_path, '*.jpg'))
# depths = glob(os.path.join(depths_path, '*.png'))
# poses = glob(os.path.join(poses_path, '*.txt'))

# Initialize containers for camera and image data
cameras_data = {}
images_data = []

for image_id, image_fn in enumerate(tqdm(images)):
    file_name = os.path.basename(image_fn)[:-4]
    pose_fn = os.path.join(poses_path, f"{file_name}.txt")
    print(file_name)

    camera_id = 1
    if camera_id not in cameras_data:
        cameras_data[camera_id] = {
            "H": 1024,
            "W": 1024,
            "intrinsics": np.array([512, 512, 512, 512]),
            "distortion": np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        }

    T = np.identity(4) # world see cam
    with open(pose_fn, 'r') as f:
        lines = f.readlines()
        T[0] = np.array(lines[0].strip().split())
        T[1] = np.array(lines[1].strip().split())
        T[2] = np.array(lines[2].strip().split())
    print(T)
    T = np.linalg.inv(T) # cam see world

    R_w = T[:3, :3]
    t_w = T[:3, 3]
    Quat_w = R.from_matrix(R_w).as_quat()

    image_name = os.path.relpath(image_fn, images_path)
    images_data.append({
        "image_id": image_id,
        "qw": Quat_w[3],
        "qx": Quat_w[0],
        "qy": Quat_w[1],
        "qz": Quat_w[2],
        "tx": t_w[0],
        "ty": t_w[1],
        "tz": t_w[2],
        "camera_id": camera_id, 
        "name": image_name
    })

colmap_dir = os.path.join(scene_path, 'colmap')
os.makedirs(colmap_dir, exist_ok=True)

print("Saved colmap model to ", colmap_dir)
# Write to cameras.txt
with open(os.path.join(colmap_dir, 'cameras.txt'), 'w') as f:

    f.write("# Camera list with one line of data per camera:\n")
    f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write("# Number of cameras: 1\n")
    for camera_id, data in cameras_data.items():
        line = f"{camera_id} PINHOLE {data['W']} {data['H']} {' '.join(map(str, data['intrinsics']))}\n"
        f.write(line)

# Write to images.txt
with open(os.path.join(colmap_dir, 'images.txt'), 'w') as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

    for image_data in images_data:
        line = f"{image_data['image_id']} {image_data['qw']} {image_data['qx']} {image_data['qy']} {image_data['qz']} {image_data['tx']} {image_data['ty']} {image_data['tz']} {image_data['camera_id']} {image_data['name']}\n"
        f.write(line)
        f.write("\n")

# Create ply file
mesh = trimesh.load(ply_path)
vertices = mesh.vertices
vertex_colors = mesh.visual.vertex_colors

# visualize ply file and camera poses


# Create an empty points3D.txt in colmap directory
with open(os.path.join(colmap_dir, 'points3D.txt'), 'w') as f:
    f.write("# 3D point list with one line of data per point:\n")
    f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    f.write(f"# Number of points: {len(vertices)}\n")
    for i in range(len(vertices)):
        line = f"{i} {vertices[i][0]} {vertices[i][1]} {vertices[i][2]} {vertex_colors[i][0]} {vertex_colors[i][1]} {vertex_colors[i][2]} 0.0 0.0\n"
        f.write(line)

with open(os.path.join(colmap_dir, 'image_list.txt'), 'w') as f:
    for image_data in images_data:
        f.write(f"{image_data['name']}\n")


    # Database file path
database_path = os.path.join(colmap_dir, 'database.db')
print("Saved database to", database_path)

# Create and connect to the database
conn = create_connection(database_path)
if conn is not None:
    create_tables(conn)

    # Start a single transaction
    conn.execute('BEGIN')

    # Example usage
    camera_id = 1
    model = 1  # Assuming 1 represents PINHOLE
    width = cameras_data[camera_id]["W"]
    height = cameras_data[camera_id]["H"]
    # import pdb; pdb.set_trace()
    params = cameras_data[camera_id]["intrinsics"].tolist()  # Convert torch tensor to list
    prior_focal_length = -1

    insert_camera(conn, camera_id, model, width, height, params, prior_focal_length)

    for image_data in images_data:

        # Image data to insert
        img_data = (image_data["image_id"], image_data["name"], camera_id, 
                    image_data["qw"], image_data["qx"], image_data["qy"], image_data["qz"], 
                    image_data["tx"], image_data["ty"], image_data["tz"])
        insert_image(conn, img_data)

    # Commit the transaction
    conn.commit()
    conn.close()
else:
    print("Error! Cannot create the database connection.")        

