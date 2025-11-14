###############################################################
### Copy Images
###############################################################
# python copy_images/preprocess_okutama.py


###############################################################
### Run SfM from scratch
###############################################################

#### Scenario3 
# COLMAP_DIR=/mnt/hdd/data/Okutama_Action/GS_data/Scenario3
# IMG_PATH=/mnt/hdd/data/Okutama_Action/GS_data/Scenario3/images

#### GS_data/Scenario4
# SCENE=1_2_2
# COLMAP_DIR=/mnt/hdd/data/Okutama_Action/GS_data/Scenario5/$SCENE
# IMG_PATH=/mnt/hdd/data/Okutama_Action/GS_data/Scenario5/$SCENE/images

##### VisDrone
SCENE=uav0000013_00000_v
COLMAP_DIR=/mnt/hdd/data/VisDrone/VisDrone-VID/GS_data/$SCENE
IMG_PATH=/mnt/hdd/data/VisDrone/VisDrone-VID/GS_data/$SCENE/images

# python process_colmap_sfm_scratch.py --colmap_dir $COLMAP_DIR \
#  --image_path $IMG_PATH 


###############################################################
### Transform SfM 
### In docker container (colmap:latest)
### THis code requires COLMAP 3.9 and pycolmap
###############################################################
# Scenario2 : 1128, 1145, 934, 951
 
# DATA_PATH=/data/Okutama_Action/GS_data/Scenario3
# # DATA_PATH=/mnt/hdd/data/Okutama_Action/GS_data/Scenario2
# python utils/get_transform_from_ref_images.py  \
#     --input_path ${DATA_PATH}/sparse/0 \
#     --out_transform_path transform.txt --idx_o 1263 \
#     --idx_x "1263,1274" --idx_y "203,180" 
#     # --rot_along_Y


# DATA_PATH=/data/Okutama_Action/GS_data/Scenario2
# # DATA_PATH=/mnt/hdd/data/Okutama_Action/GS_data/Scenario2
# python utils/get_90deg_rotation.py  \
#     --input_path ${DATA_PATH}/sparse_model_aligned \
#     --out_transform_path transform.txt --idx_o 359 \


# COLMAP_IN=/mnt/hdd/data/Okutama_Action/GS_data/Scenario3/sparse/0
# COLMAP_OUT=/mnt/hdd/data/Okutama_Action/GS_data/Scenario3/sparse_aligned
# colmap model_transformer --input_path $COLMAP_IN \
#                          --output_path $COLMAP_OUT \
#                          --transform_path ./transform.txt


# colmap model_converter --input_path $COLMAP_OUT --output_path $COLMAP_OUT/points3D.ply --output_type PLY

###############################################################
#### undistort images
###############################################################

#### Scenario3 
# COLMAP_IN=/mnt/hdd/data/Okutama_Action/GS_data/Scenario3/sparse_aligned
# UNDISTORT_OUT=/mnt/hdd/data/Okutama_Action/GS_data/Scenario3/undistorted
# IMAGE_PATH=/mnt/hdd/data/Okutama_Action/GS_data/Scenario3/images

#### GS_data/Scenario4
COLMAP_IN=$COLMAP_DIR/sparse/0
IMAGE_PATH=$IMG_PATH
UNDISTORT_OUT=$COLMAP_DIR/undistorted
echo $COLMAP_IN
echo $IMAGE_PATH
colmap image_undistorter --image_path $IMAGE_PATH \
                         --input_path $COLMAP_IN \
                         --output_path $UNDISTORT_OUT \
                         --output_type COLMAP \
                         --max_image_size 2000

#### Run MVS
# WINDOW_RADII=5
# colmap patch_match_stereo --workspace_path $UNDISTORT_OUT \
#                     --PatchMatchStereo.window_radius $WINDOW_RADII \
#                     --PatchMatchStereo.min_triangulation_angle 3.0 \
#                     --PatchMatchStereo.filter 1 \
#                     --PatchMatchStereo.geom_consistency 1 \
#                     --PatchMatchStereo.gpu_index -1 --PatchMatchStereo.num_samples 15 \
#                     --PatchMatchStereo.num_iterations 4

# mkdir -p $UNDISTORT_OUT/sparse/0
# mv $UNDISTORT_OUT/sparse/*.bin $UNDISTORT_OUT/sparse/0

# OUT_FILE=$UNDISTORT_OUT/fused.ply
# colmap stereo_fusion --workspace_path $UNDISTORT_OUT \
#                      --output_path $OUT_FILE \
#                      --input_type geometric
    
# TRIM=7 # 3 or 7
# POISSON_OUT_FILE=$UNDISTORT_OUT/meshed_trim_7.ply
# colmap poisson_mesher --input_path $OUT_FILE --output_path $POISSON_OUT_FILE --PoissonMeshing.trim $TRIM

###############################################################
### Convert colmap to nerfstudio format
### Use nerfstudio v0.3.4 / conda activate nerfstudiov0.3 / nerfstudio lib is located in third_party
### It generates depth folder (sfm depth) and transforms.json
###############################################################
# DATA_PATH=/mnt/hdd/data/Okutama_Action/GS_data/Scenario1_1_1/images
# COLMAP_MODEL_PATH=sparse/0
# OUTPUT_DIR=/mnt/hdd/data/Okutama_Action/GS_data/Scenario1_1_1
# CAMERA_TYPE=perspective

# mkdir -p $COLMAP_DIR/colmap && cd $COLMAP_DIR/colmap
# ln -s  $COLMAP_DIR/sparse ./

# ns-process-data images --data $DATA_PATH \
#     --output-dir $OUTPUT_DIR --camera_type $CAMERA_TYPE \
#     --skip-colmap --colmap-model-path $COLMAP_MODEL_PATH \
#     --skip-image-processing \
#     --use-sfm-depth --include-depth-debug


###############################################################
### Convert colmap to nerfies format
### conda activet pycolmap
### pip install numpy opencv-python pillow
### pip install git+https://github.com/google/nerfies.git@v2
### pip install tensorflow_graphics
### pip install "git+https://github.com/google/nerfies.git#egg=pycolmap&subdirectory=third_party/pycolmap"
### pip install pandas
### nerfies lib is located in third_party
###############################################################
