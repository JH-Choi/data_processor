## building-pixsfm
DATA_PATH=/data/mega_nerf_data/Mill19/building/building-pixsfm
CODE_PATH=/workspace/gaussian_splatting/data_processor

## building-pixsfm
# DATA_PATH=/data/mega_nerf_data/Mill19/rubble-pixsfm
# CODE_PATH=/workspace/gaussian_splatting/data_processor

# python convert_to_colmap.py --data_dir ${DATA_PATH} --resolution 4

# python process_colmap_sfm.py --colmap_dir ${DATA_PATH}/colmap_data \
#                              --image_path ${DATA_PATH}

python split_colmap_model.py --input_path ${DATA_PATH}/colmap_data \
                             --output_path ${DATA_PATH}/colmap_aligned \
                             --tiles "4,2,1" \
                             --transform_path ${CODE_PATH}/configs/Mill19/transform.txt \
                             --bbox_path ${CODE_PATH}/configs/Mill19/bbox.txt


## Split Point Cloud
# python split_point_cloud.py --input_file $DATA_PATH/mvs/fused_after_sor_nb16_std1.0.ply \
#                             --model_path $DATA_PATH/colmap_aligned

## Split Mesh
# python split_mesh.py --input_file /mnt/hdd$DATA_PATH/mvs/meshed_trim_7_after_sor_deci0.75.ply \
#                      --model_path /mnt/hdd$DATA_PATH/colmap_aligned
