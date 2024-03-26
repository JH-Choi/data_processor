import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import sqlite3
import struct
import shutil

import argparse

def create_connection(db_file):
    """Create a database connection to the SQLite database specified by db_file."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)
    return conn

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

def parse_mappings(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            image, pt_file = line.strip().split(',')
            mapping[image] = pt_file
    return mapping

def load_image_and_metadata_from_directory(image_dir, metadata_dir):
    rgbs = os.listdir(image_dir)
    pts = os.listdir(metadata_dir)

    ids_rgb = set([rgb.split('.')[0] for rgb in rgbs])
    ids_pt = set([pt.split('.')[0] for pt in pts])

    ids = ids_pt.intersection(ids_rgb)
    ids = sorted(list(ids))

    mapping = {}
    for id in ids:
        rgb = f"{id}.jpg"
        pt = f"{id}.pt"
        mapping[rgb] = pt
    return mapping

def load_pt_file(file_path):
    try:
        data = torch.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading .pt file: {file_path}. Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to COLMAP format')
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--resolution", type=int, default=1, help="Resolution downscale factor for the images")
    args = parser.parse_args()

    data_dir = args.data_dir
    resolution = args.resolution

    print("Processing data folder: ", data_dir)
    print("Downscale Factor: ", resolution)

    colmap_dir = os.path.join(data_dir, "colmap")
    if(os.path.exists(colmap_dir)):
        # remove directory after askting
        print("COLMAP directory '{}' already exists. Do you want to remove it? (y/n)".format(colmap_dir))
        choice = input().lower()
        if choice == 'y':
            shutil.rmtree(os.path.join(data_dir, "colmap"))

    os.makedirs(os.path.join(data_dir, "colmap"), exist_ok=True)

    # Initialize containers for camera and image data
    cameras_data = {}
    images_data = []

    # mappings_file = os.path.join(data_dir, 'mappings.txt')

    configs = ["train", "val"]
    for config in configs:
        config_folder = config

        if resolution == 1:
            rgb_folder = "rgbs"
        else:
            rgb_folder = f"rgbs_{resolution}"

        metadata_dir = os.path.join(data_dir, config_folder, 'metadata')
        rgbs_dir = os.path.join(data_dir, config_folder, rgb_folder)

        camera_resize_factor = 4.0;

        coords_path = os.path.join(data_dir, "coordinates.pt")
        coords_data = torch.load(coords_path)

        P_origin_w_drb = coords_data["origin_drb"]
        scale_factor = coords_data["pose_scale_factor"]

        mappings = load_image_and_metadata_from_directory(rgbs_dir, metadata_dir)


        for idx, (image, pt_file) in enumerate(mappings.items()):
            metadata_path = os.path.join(metadata_dir, pt_file)
            pt_data = load_pt_file(metadata_path)

            assert pt_data is not None

            camera_id = 1
            if camera_id not in cameras_data:
                cameras_data[camera_id] = {
                    "H": pt_data['H']/camera_resize_factor,
                    "W": pt_data['W']/camera_resize_factor,
                    "intrinsics": pt_data['intrinsics']/camera_resize_factor,
                    "distortion": pt_data['distortion']/camera_resize_factor
                }

            image_id = int(image.split('.')[0])

            # Extract extrinsics (rotation and translation)
            c2w = pt_data['c2w']

            # Convert rotation matrix to quaternion (if required by COLMAP)
            R_w_rub = c2w[:3, :3].numpy()
            P_w_drb = (scale_factor * c2w[:3, 3] - P_origin_w_drb).numpy()

            R_drb_rub = np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ])
            R_w_drb = R_w_rub @ R_drb_rub.T

            R_drb_w = R_w_drb.T
            P_drb_w = (R_drb_w @ (-P_w_drb))


            R_rdf_drb = np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, -1],
            ])

            R_rub_w = R_rdf_drb @ R_drb_w
            P_rub_w = R_rdf_drb @ P_drb_w

            Quat_rub_w = R.from_matrix(R_rub_w).as_quat()

            image_name = os.path.relpath(os.path.join(rgbs_dir, image), data_dir)
            images_data.append({
                "image_id": image_id,
                "qw": Quat_rub_w[3],
                "qx": Quat_rub_w[0],
                "qy": Quat_rub_w[1],
                "qz": Quat_rub_w[2],
                "tx": P_rub_w[0],
                "ty": P_rub_w[1],
                "tz": P_rub_w[2],
                "camera_id": camera_id, 
                "name": image_name
            })

    print("Saved colmap model to ", colmap_dir)
    # Write to cameras.txt
    with open(os.path.join(colmap_dir, 'cameras.txt'), 'w') as f:

        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        for camera_id, data in cameras_data.items():
            line = f"{camera_id} PINHOLE {data['W']} {data['H']} {' '.join(map(str, data['intrinsics'].numpy()))}\n"
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

    # Create an empty points3D.txt in colmap directory
    with open(os.path.join(colmap_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")

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

if __name__ == "__main__":
    main()

