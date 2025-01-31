
# conda activate nerf2mesh

# SCENE_LIST=('Drone1_Noon_1_2_2' 'Drone1_Noon_1_2_4' 'Drone1_Noon_1_2_6' 'Drone1_Noon_1_2_8' 'Drone1_Noon_1_2_9' 'Drone1_Noon_1_2_11' 'Drone2_Noon_2_2_2' 'Drone2_Noon_2_2_4' 'Drone2_Noon_2_2_8' 'Drone2_Noon_2_2_9' 'Drone2_Noon_2_2_11')
SCENE_LIST=('Drone1_Morning_1_1_1' 'Drone1_Morning_1_1_2' 'Drone1_Morning_1_1_3' 'Drone1_Morning_1_1_4' 'Drone1_Morning_1_1_7' 'Drone1_Morning_1_1_10' 'Drone1_Morning_1_1_11' 'Drone2_Morning_2_1_1' 'Drone2_Morning_2_1_2' 'Drone2_Morning_2_1_3' 'Drone2_Morning_2_1_4' 'Drone2_Morning_2_1_6' 'Drone2_Morning_2_1_7' 'Drone2_Morning_2_1_10' )

for split in "${SCENE_LIST[@]}"
do
    echo $split
    # input_model=/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/sparse/0
    input_model=/mnt/hdd/data/Okutama_Action/GS_data/Scenario3/undistorted/sparse/0
    input_format=.bin
    # split=Drone1_Noon_1_2_8
    # output_model=/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/sparse_split/$split/0
    output_model=/mnt/hdd/data/Okutama_Action/GS_data/Scenario3/undistorted/sparse_split/$split/0
    output_format=.bin

    mkdir -p $output_model

    python remove_images_from_colmap.py --input_model $input_model \
        --input_format $input_format --split $split \
        --output_model $output_model --output_format $output_format

done