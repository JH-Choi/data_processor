

SCENE_LIST=('1_2_9' '1_2_11' '2_2_2' '2_2_4' '2_2_8' '2_2_9' '2_2_11' '1_1_1' '1_1_2' '1_1_3' '1_1_4' '1_1_7' '1_1_10' '1_1_11' '2_1_1' '2_1_2' '2_1_3' '2_1_4' '2_1_6' '2_1_7' '2_1_10')

for SCENE in "${SCENE_LIST[@]}"
do
    echo $SCENE

    COLMAP_DIR=/mnt/hdd/data/Okutama_Action/GS_data/Scenario4/$SCENE
    IMG_PATH=/mnt/hdd/data/Okutama_Action/GS_data/Scenario4/$SCENE/images
    echo $COLMAP_DIR
    echo $IMG_PATH

    python process_colmap_sfm_scratch.py --colmap_dir $COLMAP_DIR \
    --image_path $IMG_PATH 

    COLMAP_IN=$COLMAP_DIR/sparse/0
    IMAGE_PATH=$IMG_PATH
    UNDISTORT_OUT=$COLMAP_DIR/undistorted
    colmap image_undistorter --image_path $IMAGE_PATH \
                            --input_path $COLMAP_IN \
                            --output_path $UNDISTORT_OUT \
                            --output_type COLMAP \
                            --max_image_size 2000

done        
