# cd ../converter/


# docker 
# nerfstudio container

# pip install pycolmap-0.0.1 
# cd third_party/nerfies/third_party/pycolmap
# pip install -e .
# pip install tensorflow==2.8.0
# pip install numpy==1.26.4

# conda activate pycolmap
# ROOT_DIR=/data/VisDrone/VisDrone-VID/GS_data/uav0000084_00000_v/preprocessed
# python ./converter/colmap_to_nerfies_vggt.py --root_dir $ROOT_DIR



# SCENE_NAMES=("Noon_1_2_2" "Noon_1_2_6" "Noon_1_2_8" "Noon_1_2_9")
# for SCENE_NAME in ${SCENE_NAMES[@]}; do 
#     DATA_PATH=/data/Okutama_Action/GS_data/Scenario2/preprocessed/$SCENE_NAME
#     python ./converter/colmap_to_nerfies_vggt.py --root_dir $DATA_PATH

# done


# SCENE_NAMES=("40_GND_P1" "40_VGT_P1" "50_GND_P1" "50_RD_P1")

# for SCENE_NAME in ${SCENE_NAMES[@]}; do 
#     DATA_PATH=/data/Manipal-UAV-Person/train/GS_data/$SCENE_NAME/preprocessed
#     python ./converter/colmap_to_nerfies_vggt.py --root_dir $DATA_PATH
# done



SCENE_NAMES=("uav0000013_00000_v" "uav0000079_00480_v" "uav0000084_00000_v" "uav0000099_02109_v")

for SCENE_NAME in ${SCENE_NAMES[@]}; do 
    DATA_PATH=/data/VisDrone/VisDrone-VID/GS_data/$SCENE_NAME/preprocessed
    python ./converter/colmap_to_nerfies_vggt.py --root_dir $DATA_PATH
done

