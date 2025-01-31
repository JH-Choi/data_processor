# Data Preprocessing

## Okutama-Action 
Okutama does not have GT poses

1. Create training scenario 
```bash
python copy_images/preprocess_okutama.py
```

2. Run SfM from scratch
```bash
COLMAP_DIR=/mnt/hdd/data/Okutama_Action/GS_data/Scenario2
IMG_PATH=/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/images

python process_colmap_sfm_scratch.py --colmap_dir $COLMAP_DIR \
 --image_path $IMG_PATH 
```

3. Transform SfM Model


## Mega-NeRF 



## Etc
- Convert colmap bin file to ply file 
```bash
colmap model_converter --intput_path 0 --output_path points3D.ply --output_type PLY
```