
###############################################################
### Transform SfM 
### In docker container (colmap:latest)
### THis code requires COLMAP 3.9 and pycolmap
###############################################################
# DATA_PATH=/data/Citygaussian_data/colmap_results/rubble/train/

# python utils/get_transform_from_ref_images.py  \
#     --input_path ${DATA_PATH}/sparse/0 \
#     --out_transform_path ./transform.txt --idx_o 796 \
#     --idx_x "796,1290" --idx_y "1657,1636"


OUT_PATH=/mnt/hdd/data/Citygaussian_data/colmap_results/rubble/train/sparse_aligned
# mkdir -p $OUT_PATH/0
# mv transform.txt $OUT_PATH


COLMAP_IN=/mnt/hdd/data/Citygaussian_data/colmap_results/rubble/train/sparse
COLMAP_OUT=$OUT_PATH
# colmap model_transformer --input_path $COLMAP_IN \
#                          --output_path $COLMAP_OUT/0 \
#                          --transform_path $OUT_PATH/transform.txt

# colmap model_converter --input_path $COLMAP_OUT/0 --output_path $COLMAP_OUT/0/points3D.ply --output_type PLY


#### undistort images
RAW_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/rubble-pixsfm/train/rgbs
COLMAP_IN=/mnt/hdd/data/Citygaussian_data/colmap_results/rubble/train/sparse_aligned/0
UNDISTORT_OUT=/mnt/hdd/data/Citygaussian_data/colmap_results/rubble/train/sparse_distorted
# mkdir -p $UNDISTORT_OUT
# colmap image_undistorter --image_path $RAW_PATH \
#                          --input_path $COLMAP_IN \
#                          --output_path $UNDISTORT_OUT \
#                          --output_type COLMAP \
#                          --max_image_size 2000

# mkdir -p $UNDISTORT_OUT/sparse/0
# mv $UNDISTORT_OUT/sparse/*.bin $UNDISTORT_OUT/sparse/0

# colmap model_converter --input_path $UNDISTORT_OUT/sparse/0 --output_path $UNDISTORT_OUT/sparse/0/points3D.ply --output_type PLY


###############################################################
### Transform SfM for validation set
###############################################################
# TRANS_FILE=/mnt/hdd/data/Citygaussian_data/colmap_results/rubble/train/sparse_aligned/transform.txt
# OUT_PATH=/mnt/hdd/data/Citygaussian_data/colmap_results/rubble/val/sparse_aligned
# mkdir -p $OUT_PATH/0
# COLMAP_IN=/mnt/hdd/data/Citygaussian_data/colmap_results/rubble/val/sparse
# COLMAP_OUT=$OUT_PATH
# colmap model_transformer --input_path $COLMAP_IN/0 \
#                          --output_path $COLMAP_OUT/0 \
#                          --transform_path $TRANS_FILE

