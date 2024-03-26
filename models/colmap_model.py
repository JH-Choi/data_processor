
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

class Image:
    def __init__(self, image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name):
        self.image_id = image_id
        self.q_cw = np.array([qx, qy, qz, qw])
        self.t_cw = np.array([tx, ty, tz])

        self.camera_id = camera_id
        self.name = name
        self.keypoints = []  # This will hold tuples of (u, v, d, pts_rgb)

    @property
    def R_cw(self):
        return R.from_quat(self.q_cw).as_matrix()

    @property
    def T_wc(self):
        return np.linalg.inv(np.block([[self.R_cw, self.t_cw.reshape(-1, 1)], [0, 0, 0, 1]]))

class Point3D:
    def __init__(self, point_id, x, y, z, r, g, b, e=None, track=[]):
        self.point_id = point_id
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.e = e
        self.track = track  # This will hold image_ids where the point is visible

class Camera:
    def __init__(self, camera_id, model, width, height, params):
        self.camera_id = camera_id
        self.model = model
        assert model in ['PINHOLE']
        self.width = width
        self.height = height
        self.params = params  # This can be a list or tuple of parameters depending on the camera model

    @property
    def fx(self):
        return self.params[0]

    @property
    def fy(self):
        return self.params[1]

    @property
    def cx(self):
        return self.params[2]

    @property
    def cy(self):
        return self.params[3]

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx],
                     [0, self.fy, self.cy],
                     [0, 0, 1]])


def read_colmap_model(colmap_dir):
    image_file_path = colmap_dir + "/images.txt"
    points_file_path = colmap_dir + "/points3D.txt"
    camera_file_path = colmap_dir + "/cameras.txt"

    cameras = {}
    print("Reading cameras")
    with open(camera_file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(float(parts[2]))
            height = int(float(parts[3]))
            params = list(map(float, parts[4:]))  # Convert all remaining parts to floats
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    assert(len(cameras) == 1)
    camera = list(cameras.values())[0]

    points3d = {}
    print("Reading points3D")
    with open(points_file_path, 'r') as file:
        # Directly filter out comment lines and strip whitespace for processing
        lines = [line.strip() for line in file if not line.startswith('#')]

        for line in tqdm(lines, desc="Reading 3D points"):
            parts = line.split()

            point_id = int(parts[0])
            # Assuming Point3D's constructor can take all needed parameters directly
            # and that it has an attribute 'track' initialized to an empty list within the constructor
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            e = float(parts[7])
            track = [(int(parts[i]), int(parts[i+1])) for i in range(8, len(parts), 2)]
            
            points3d[point_id] = Point3D(point_id, x, y, z, r, g, b, e, track)

    images = {}
    print("Reading images")
    with open(image_file_path, 'r') as file:
        # Directly filter out comment lines and strip whitespace for processing
        lines = [line.strip() for line in file if not line.startswith('#')]
        assert len(lines) % 2 == 0, "The number of lines in the images file should be even"

        for i in tqdm(range(0, len(lines), 2), desc="Reading images"):
            image_parts = lines[i].split()
            keypoint_parts = lines[i + 1].split()

            # Extracting information directly
            image_id = int(image_parts[0])
            qw, qx, qy, qz, tx, ty, tz = map(float, image_parts[1:8])
            camera_id, name = int(image_parts[8]), image_parts[9]
            images[image_id] = Image(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)

            image = images[image_id]

            for j in range(0, len(keypoint_parts), 3):
                if int(keypoint_parts[j + 2]) != -1:
                    point3d = points3d[int(keypoint_parts[j + 2])]
                    uv_hom = np.array([float(keypoint_parts[j]), float(keypoint_parts[j + 1]), 1.0])
                    xy_hom = np.linalg.inv(camera.K) @ uv_hom
                    x, y = xy_hom[:2] / xy_hom[2]
                    pts_w = np.array([point3d.x, point3d.y, point3d.z])
                    pts_c = image.R_cw @ pts_w + image.t_cw
                    d = pts_c[2]
                    r, g, b = (point3d.r, point3d.g, point3d.b)
                    images[image_id].keypoints.append((x, y, d, r, g, b))

    return {"images":images, "points3D": points3d, "cameras":cameras}

