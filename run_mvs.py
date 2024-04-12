from utils.database import COLMAPDatabase
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d
import imageio
import subprocess
import argparse
import os
import pdb 

def bash_run(cmd):
    # local install of colmap
    env = os.environ.copy()
    # env['LD_LIBRARY_PATH'] = '/home/jaehoon/code/colmap/build/__install__/lib'
    # env['LD_LIBRARY_PATH'] = '/usr/local/bin/colmap/build/__install__/lib'

    # colmap_bin = '/home/jaehoon/code/colmap/build/__install__/bin/colmap'
    colmap_bin = '/usr/local/bin/colmap'
    cmd = colmap_bin + ' ' + cmd
    print('\nRunning cmd: ', cmd)

    subprocess.check_call(['/bin/bash', '-c', cmd], env=env)

gpu_index = '-1'

def run_point_triangulation(img_dir, db_file, out_dir):
    print('Running point triangulation...')

    # triangulate points
    cmd = ' point_triangulator  --database_path {} \
                                --image_path {} \
                                --input_path {} \
                                --output_path {} \
                                --Mapper.tri_ignore_two_view_tracks 1'.format(db_file, img_dir, out_dir, out_dir)
    bash_run(cmd)


def prepare_mvs(img_dir, sfm_dir, mvs_dir):
    if not os.path.exists(mvs_dir):
        os.mkdir(mvs_dir)

    sparse_symlink = os.path.join(mvs_dir, 'sparse')
    if os.path.exists(sparse_symlink):
       os.unlink(sparse_symlink)
    # os.symlink(os.path.relpath(sfm_dir, mvs_dir), sparse_symlink)
    os.symlink(sfm_dir, sparse_symlink)

    # prepare stereo directory
    stereo_dir = os.path.join(mvs_dir, 'stereo')
    for subdir in [stereo_dir, 
                   os.path.join(stereo_dir, 'depth_maps'),
                   os.path.join(stereo_dir, 'normal_maps'),
                   os.path.join(stereo_dir, 'consistency_graphs')]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    # # write patch-match.cfg and fusion.cfg
    # image_names = sorted(os.listdir(os.path.join(mvs_dir, 'images')))

    # with open(os.path.join(stereo_dir, 'patch-match.cfg'), 'w') as fp:
    #     for img_name in image_names:
    #         fp.write(img_name + '\n__auto__, 20\n')
    #         
    #         # use all images
    #         # fp.write(img_name + '\n__all__\n')

    #         # randomly choose 20 images
    #         # from random import shuffle
    #         # candi_src_images = [x for x in image_names if x != img_name]
    #         # shuffle(candi_src_images)
    #         # max_src_images = 10
    #         # fp.write(img_name + '\n' + ', '.join(candi_src_images[:max_src_images]) + '\n')

    # with open(os.path.join(stereo_dir, 'fusion.cfg'), 'w') as fp:
    #     for img_name in image_names:
    #         fp.write(img_name + '\n')

    
    # run_undistort_images(images_symlink, sparse_symlink, mvs_dir)
    run_undistort_images(img_dir, sparse_symlink, mvs_dir)

def run_undistort_images(img_dir, input_dir, out_dir):
    print('Running image undistorter...')

    # triangulate points
    cmd = ' image_undistorter  --image_path {} \
                                      --input_path {} \
                                      --output_path {}'.format(img_dir, input_dir, out_dir)
    bash_run(cmd)


def run_point_triangulation(img_dir, db_file, out_dir):
    print('Running point triangulation...')

    # triangulate points
    cmd = ' point_triangulator  --database_path {} \
                                      --image_path {} \
                                      --input_path {} \
                                      --output_path {} \
                                      --Mapper.ba_refine_focal_length 1 \
                                      --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0'.format(db_file, img_dir, out_dir, out_dir)
    bash_run(cmd)


def run_photometric_mvs(mvs_dir, window_radius):
    print('Running photometric MVS...')

    cmd = ' patch_match_stereo --workspace_path {} \
                    --PatchMatchStereo.window_radius {} \
                    --PatchMatchStereo.min_triangulation_angle 3.0 \
                    --PatchMatchStereo.filter 1 \
                    --PatchMatchStereo.geom_consistency 1 \
                    --PatchMatchStereo.gpu_index={} \
                    --PatchMatchStereo.num_samples 15 \
                    --PatchMatchStereo.num_iterations 4'.format(mvs_dir,
                                                                 window_radius, gpu_index)

    bash_run(cmd)


def run_fuse(mvs_dir, out_ply):
    print('Running depth fusion...')

    cmd = ' stereo_fusion --workspace_path {} \
                         --output_path {} \
                         --input_type geometric'.format(mvs_dir, out_ply)
    bash_run(cmd)

def run_SOR(pcd_file, out_file, voxel_size=0.02, nb_neighbors=16, std_ratio=1.0):
    print('Running statistical outlier removal...')

    pcd = o3d.io.read_point_cloud(pcd_file) # 30990087 points

    print(f"Downsample the point cloud with a voxel of {voxel_size}")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    print("Statistical oulier removal")
    filter_pcd, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors, \
                                                                std_ratio, \
                                                                print_progress=True)

    o3d.io.write_point_cloud(out_file, filter_pcd)

def run_possion_mesher(in_ply, out_ply, trim):
    print('Running possion mesher...')

    cmd = ' poisson_mesher \
            --input_path {} \
            --output_path {} \
            --PoissonMeshing.trim {}'.format(in_ply, out_ply, trim)

    bash_run(cmd)

def run_delaunay_mesher(in_ply, out_ply):
    print('Running delaunay mesher...')

    cmd = ' delaunay_mesher \
            --input_path {} \
            --output_path {}'.format(in_ply, out_ply)

    bash_run(cmd)


def main(img_dir, sfm_dir, out_dir):
    mvs_dir = out_dir
    if not os.path.exists(mvs_dir):
        os.mkdir(mvs_dir)

    # run_sift_matching(img_dir, img_txt_file, db_file)

    # root_folder = '/data/Okutama_Action/Chris_data/1.1.1/training_set/train'
    # run_point_triangulation(root_folder, db_file, sfm_dir)

    # # optional
    # # run_global_ba(sfm_dir, sfm_dir)

    # img_dir = os.path.join(root_folder, 'images')
    prepare_mvs(img_dir, sfm_dir, mvs_dir)
    run_photometric_mvs(mvs_dir, window_radius=5)

    out_ply_before_sor = os.path.join(mvs_dir, 'fused.ply')
    run_fuse(mvs_dir, out_ply_before_sor)

    out_ply_after_sor = os.path.join(mvs_dir, 'fused_after_sor.ply')
    run_SOR(out_ply_before_sor, out_ply_after_sor)
    run_fuse(mvs_dir, out_ply_after_sor)

    # out_poisson_mesh_ply = os.path.join(mvs_dir, 'meshed_trim_3.ply')
    # run_possion_mesher(out_ply, out_poisson_mesh_ply, trim=3)
    
    out_poisson_mesh_ply = os.path.join(mvs_dir, 'meshed_trim_7_after_sor.ply')
    run_possion_mesher(out_ply_after_sor, out_poisson_mesh_ply, trim=7)

    # out_delaunay_mesh_ply = os.path.join(mvs_dir, 'delaunay_mesh.ply')
    # run_delaunay_mesher(mvs_dir, out_delaunay_mesh_ply)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process COLMAP MVS reconstruction')
    parser.add_argument("--root_folder", type=str, help="Root folder of the dataset")
    args = parser.parse_args()

    root_folder = args.root_folder
    # img_dir = os.path.join(root_folder, 'images')
    # db_file = os.path.join(root_folder, 'colmap/database.db')
    sfm_dir = os.path.join(root_folder, 'colmap_aligned')
    out_dir = os.path.join(root_folder, 'mvs')

    main(img_dir=root_folder, sfm_dir=sfm_dir, out_dir=out_dir) 

