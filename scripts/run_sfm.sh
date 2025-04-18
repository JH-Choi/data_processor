## building-pixsfm
DATA_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm
# DATA_PATH=/mnt/hdd/data/Okutama_Action/Yonghan_data
CODE_PATH=/workspace/gaussian_splatting/data_processor

## building-pixsfm
# DATA_PATH=/data/mega_nerf_data/Mill19/rubble-pixsfm
# CODE_PATH=/workspace/gaussian_splatting/data_processor

# python convert_to_colmap.py --data_dir ${DATA_PATH} --resolution 4

# python process_colmap_sfm.py --colmap_dir ${DATA_PATH}/colmap_data \
#                              --image_path ${DATA_PATH}

# python split_colmap_model.py --input_path ${DATA_PATH}/colmap_data \
#                              --output_path ${DATA_PATH}/colmap_aligned \
#                              --tiles "4,2,1" \
#                              --transform_path ${CODE_PATH}/configs/Mill19/transform.txt \
#                              --bbox_path ${CODE_PATH}/configs/Mill19/bbox.txt


## Split Point Cloud
# python split_point_cloud.py --input_file $DATA_PATH/mvs/fused_after_sor_nb16_std1.0.ply \
#                             --model_path $DATA_PATH/colmap_aligned

## Split Mesh
# python split_mesh.py --input_file /mnt/hdd$DATA_PATH/mvs/meshed_trim_7_after_sor_deci0.75.ply \
#                      --model_path /mnt/hdd$DATA_PATH/colmap_aligned

## Only transform colmap model
# python3.10 transform_colmap_model.py --input_path $DATA_PATH/colmap_mvs \
#                                      --output_path $DATA_PATH/colmap_mvs/total \
#                                      --transform_path $DATA_PATH/colmap_mvs/transform.txt 
# colmap model_converter --input_path $DATA_PATH/colmap_mvs/total --output_path $DATA_PATH/colmap_mvs/total/Points3D.ply --output_type PLY


# Transform 3D model
# python3.10 transform_3Dmodel.py --input_file $DATA_PATH/colmap_aligned/points3D.ply \
#                                 --output_file $DATA_PATH/colmap_aligned/points3D.ply \
#                                 --transform_path $DATA_PATH/colmap_aligned/transform.txt \
#                                 --Dtype pcd

python3.10 transform_3Dmodel.py --input_file $DATA_PATH/colmap_aligned/mesh_deci0.75.ply \
                                --output_file $DATA_PATH/colmap_aligned/mesh_deci0.75.ply \
                                --transform_path $DATA_PATH/colmap_aligned/transform.txt \
                                --Dtype mesh


# python3.10 transform_3Dmodel.py --input_file $DATA_PATH/PoissonMeshes/fused_sor_seg_mesh_lod11.ply \
#                                 --output_file $DATA_PATH/PoissonMeshes/fused_sor_seg_mseh_lod11_transformed.ply \
#                                 --transform_path $DATA_PATH/transform.txt \
#                                 --Dtype mesh

