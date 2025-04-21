# cd ../converter/


# docker 
# nerfstudio container

# pip install pycolmap-0.0.1 
# cd third_party/nerfies/third_party/pycolmap
# pip install -e .
# pip install tensorflow==2.8.0
# pip install numpy==1.26.4

# conda activate pycolmap
ROOT_DIR=/data/VisDrone/VisDrone-VID/GS_data/uav0000084_00000_v/preprocessed
python ./converter/colmap_to_nerfies_vggt.py --root_dir $ROOT_DIR