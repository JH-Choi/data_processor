import argparse
import numpy as np
import cv2

import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d


def split_name(f):
    return os.path.splitext(os.path.basename(f))[0].split('_')[0]

class DepthImage:
    @property
    def cos_depth_map(self):

        uv_map = np.array([[[i, j, 1] for i in range(self.depth_image.shape[1])] for j in range(self.depth_image.shape[0])])
        uv_map = uv_map[..., np.newaxis]

        inv_K = np.linalg.inv(self.K)
        inv_K = inv_K.reshape(1, 1, 3, 3)

        xy_hom_map = np.linalg.inv(self.K) @ uv_map
        norm_map = np.linalg.norm(xy_hom_map[:,:,:2, -1], axis=-1)

        cos_map = 1 / np.sqrt(norm_map**2 + 1)

        return self.depth_image * cos_map





    def __init__(self, depth_path, fx, fy, cx, cy, scale_factor=1):
        self.depth_path = depth_path
        self.fx = fx / scale_factor
        self.fy = fy / scale_factor
        self.cx = cx / scale_factor
        self.cy = cy / scale_factor
        self.depth_image = np.load(depth_path)
        self.is_metric = False

    # def crop_center_2d(self, crop_height, crop_width):
    #     h, w = self.depth_image.shape
    #     start_x = w // 2 - crop_width // 2
    #     start_y = h // 2 - crop_height // 2
    #     self.depth_image = self.depth_image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    #     self.cx = self.cx - start_x
    #     self.cy = self.cy - start_y

    # def resize(self, new_height, new_width):
    #     self.fx = self.fx * new_width / self.depth_image.shape[1]
    #     self.fy = self.fy * new_height / self.depth_image.shape[0]
    #     self.cx = self.cx * new_width / self.depth_image.shape[1]
    #     self.cy = self.cy * new_height / self.depth_image.shape[0]
    #     self.depth_image = cv2.resize(self.depth_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    def save_to(self, path, post_fix=""):
        if os.path.isdir(path):
            depth_name = split_name(self.depth_path) + "_" + post_fix + ".npy"
            np.save(os.path.join(path, depth_name), self.depth_image)
        else:
            np.save(path, self.depth_image)

    @property
    def shape(self):
        return self.depth_image.shape

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


def estimate_scale_from_median(ref_depth, src_depth):
    ref_median = np.median(ref_depth)
    src_median = np.median(src_depth)

    scale = ref_median / src_median
    return scale

def adjust_depth_scale(src_depth_image, ref_depth_data, num_min_matches=10, debug=False):

    num_data = ref_depth_data.shape[0]
    if num_data <= num_min_matches:
        return False

    xy = ref_depth_data[:,0:2]
    rgb = ref_depth_data[:,3:6]

    xy_hom = np.hstack((xy, np.ones((num_data, 1))))
    uv_hom = (src_depth_image.K @ xy_hom.T).T

    indices = np.where((uv_hom[:,0] >= 0) & (uv_hom[:,0] <= src_depth_image.shape[1] - 1) 
                     & (uv_hom[:,1] >= 0) & (uv_hom[:,1] <= src_depth_image.shape[0] - 1))[0]

    uv = np.round(uv_hom[indices, 0:2]).astype(int)

    ref_depth = ref_depth_data[indices, 2]
    src_depth = src_depth_image.depth_image[uv[:,1], uv[:,0]]

    scale = estimate_scale_from_median(ref_depth, src_depth)

    assert(src_depth_image.is_metric == False), "The depth image is already in metric scale"
    src_depth_image.depth_image = src_depth_image.depth_image * scale
    src_depth_image.is_metric = True


    if debug:
        # Create histograms
        src_depth_adjusted = src_depth * scale


        # Find the global minimum and maximum depth values across all datasets
        global_min_depth = min(ref_depth.min(), src_depth.min(), (src_depth * scale).min())
        global_max_depth = max(ref_depth.max(), src_depth.max(), (src_depth * scale).max())

        # Define the bin edges so that all histograms will use the same bins
        bin_edges = np.linspace(global_min_depth, global_max_depth, num=120+1)  # 120 bins

        # Create histograms using the same bin edges
        plt.figure(figsize=(30, 18))
        plt.hist(ref_depth, bins=bin_edges, alpha=0.5, label='Ref List')
        plt.hist(src_depth, bins=bin_edges, alpha=0.5, label='Src List')
        plt.hist(src_depth_adjusted, bins=bin_edges, alpha=0.5, label='Adj Src List')

        # plt.figure(figsize=(10, 6))  # Adjust the size of the plot as needed
        # plt.hist(ref_depth, bins=40, alpha=0.5, label='Ref List')  # Adjust the number of bins as needed
        # plt.hist(src_depth, bins=40, alpha=0.5, label='Src List')  # Adjust the number of bins as needed
        # plt.hist(src_depth_adjusted, bins=40, alpha=0.5, label='Adj Src List')  # Adjust the number of bins as needed

        # Adding title and labels
        plt.title('Comparison of Depth Values')
        plt.xlabel('Depth Value')
        plt.ylabel('Frequency')

        # Show legend
        plt.legend()

        # Show plot
        # plt.show()


        depth_image = src_depth_image.depth_image
        debug_scale = 65536.0 / depth_image.max()
        norm_depth_image = depth_image * debug_scale
        intrinsics = o3d.camera.PinholeCameraIntrinsic(src_depth_image.shape[1], src_depth_image.shape[0], src_depth_image.fx, src_depth_image.fy, src_depth_image.cx, src_depth_image.cy)

        o3d_depth_image = o3d.geometry.Image(norm_depth_image.astype(np.uint16))

        color_image = cv2.imread(src_depth_image.depth_path.replace("_depth", "_rgb").replace(".npy", ".png"))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image.fill(128)
        o3d_color_image = o3d.geometry.Image(color_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color_image, o3d_depth_image, depth_scale=1.0, depth_trunc=65536.0, convert_rgb_to_intensity=False)

        pcd_adj = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

        pts_xyz = debug_scale * xy_hom[indices] * ref_depth[..., np.newaxis]
        pts_rgb = rgb[indices] / 255.0

        pcd_ref = o3d.geometry.PointCloud()

        pcd_ref.points = o3d.utility.Vector3dVector(pts_xyz)
        pcd_ref.colors = o3d.utility.Vector3dVector(pts_rgb)


        pcds = [pcd_ref, pcd_adj]

        o3d.visualization.draw_geometries(pcds)



    return True


def crop_center_2d(arr, crop_height, crop_width):
    h, w = arr.shape
    start_x = w // 2 - crop_width // 2
    start_y = h // 2 - crop_height // 2
    return arr[start_y:start_y + crop_height, start_x:start_x + crop_width]

def main():
    parser = argparse.ArgumentParser(description="Estimate the scale factor between two depth images")
    parser.add_argument("--src_depth_image", required=True, help="Path to the .npy source depth image file or directory")
    parser.add_argument("--src_fx", type=float, required=True, help="Source camera focal length in x direction")
    parser.add_argument("--src_fy", type=float, required=True, help="Source camera focal length in y direction")
    parser.add_argument("--src_cx", type=float, required=True, help="Source camera principal point in x direction")
    parser.add_argument("--src_cy", type=float, required=True, help="Source camera principal point in y direction")
    parser.add_argument("--src_scale_factor", type=int, default=1, help="Division scale factor that will be applied to the given intrinsic parameters")
    parser.add_argument("--tgt_depth_path", required=True, help="Path to save the .npy metric depth image file or directory")
    parser.add_argument("--ref_depth_data", required=True, help="Path to the .npy reference sparse depth data file or directory")
    parser.add_argument("--num_min_matches", type=int, default=10, help="Minimum number of matches to adjust the scale")
    parser.add_argument("--debug", required=False, default=False, help="Debug mode")
    args = parser.parse_args()

    src_depth_image_paths = []
    ref_depth_data_paths = []

    if os.path.isdir(args.src_depth_image):
        assert os.path.isdir(args.ref_depth_data), "If the source depth image is a directory, the reference depth data should also be a directory"

        if not os.path.exists(args.tgt_depth_path):
            os.makedirs(args.tgt_depth_path)
        assert os.path.isdir(args.tgt_depth_path), "If the source depth image is a directory, the target depth image should also be a directory"

        src_depth_image_paths = [os.path.join(args.src_depth_image, f) for f in os.listdir(args.src_depth_image) if f.endswith('.npy')]
        src_depth_names = [split_name(f) for f in src_depth_image_paths]

        ref_depth_data_paths = [os.path.join(args.ref_depth_data, f) for f in os.listdir(args.ref_depth_data) if f.endswith('.npy')]
        ref_depth_names = [split_name(f) for f in ref_depth_data_paths]

        shared_depth_names = list(set(src_depth_names) & set(ref_depth_names))

        print("The Number of Source Depth Images: ", len(src_depth_image_paths))
        print("The Number of Reference Depth Data: ", len(ref_depth_data_paths))
        print("The Number of Shared Depth Names: ", len(shared_depth_names))


        ref_depth_data_paths = sorted([f for f in ref_depth_data_paths if split_name(f) in shared_depth_names])
        src_depth_image_paths = sorted([f for f in src_depth_image_paths if split_name(f) in shared_depth_names])

    else:
        assert os.path.isfile(args.ref_depth_data), "If the source depth image is a file, the reference depth data should also be a file"
        assert not os.path.isdir(args.tgt_depth_path), "If the source depth image is a file, the target depth image should also be a file"

        src_depth_image_paths.append(args.src_depth_image)
        ref_depth_data_paths.append(args.ref_depth_data)

    src_depth_images = []
    ref_depth_mats = []

    for src_depth_image_path, ref_depth_data_path in tqdm(zip(src_depth_image_paths, ref_depth_data_paths), total=len(src_depth_image_paths), desc="Reading Depth Images and Depth Data"):
        assert split_name(src_depth_image_path) == split_name(ref_depth_data_path), "The source depth image and reference depth data should have the same name"
        src_depth_image = DepthImage(src_depth_image_path, args.src_fx, args.src_fy, args.src_cx, args.src_cy, args.src_scale_factor)
        ref_depth_mats.append(np.load(ref_depth_data_path))
        src_depth_images.append(src_depth_image)

    num_adjusted = 0
    for i in tqdm(range(len(src_depth_images)), total=len(src_depth_images), desc="Adjusts scales of depth images"):
        if adjust_depth_scale(src_depth_images[i], ref_depth_mats[i], num_min_matches=args.num_min_matches, debug=args.debug):
            num_adjusted += 1
    print("The number of adjusted depth images: ", num_adjusted)

    num_saved = 0
    for i in tqdm(range(len(src_depth_images)), total=len(src_depth_images), desc="Saving depth images"):
        if src_depth_images[i].is_metric == True:
            src_depth_images[i].save_to(args.tgt_depth_path, post_fix="metric")
            num_saved += 1

    print("The number of saved depth images: ", num_saved)

if __name__ == "__main__":
    main()
