# Data Preprocessing


## Transform COLMAP SfM Model
There are two ways to transform the COLMAP SfM model.

### Method1: Manually Find transformation matrix  
In Docker container, you can use the following command to find the transformation matrix.

```bash
python utils/get_transform_from_ref_images.py --input_path ${DATA_PATH}/sparse/0 --out_transform_path transform.txt --idx_o=941 --ids_x "941, 1034", --ids_y "1939, 1915"

COLMAP_IN=${DATA_PATH}/sparse/0
COLMAP_OUT=${DATA_PATH}/sparse_aligned

colmap model_transformer --input_path $COLMAP_IN --output_path $COLMAP_OUT --transform_path ./transform.txt

colmap model_converter --input_path $COLMAP_OUT --output_path $COLMAP_OUT/points3D.ply --output_type PLY
```


### Method2: Use COLMAP module  
model_orientation_aligner: Align the coordinate axis of a model using a Manhattan world assumption.

```bash
colmap model_orientation_aligner --image_path ${DATA_PATH}/images --input_path ${DATA_PATH}/sparse/0 --output_path ${DATA_PATH}/sparse_aligned 

# In Docker container, you can get transform.txt
python utils/get_90deg_rotation.py  --input_path ${DATA_PATH}/sparse_aligned --out_transform_path transform.txt --idx_o 359 

COLMAP_IN=${DATA_PATH}/sparse_aligned
COLMAP_OUT=${DATA_PATH}/sparse_transformed

colmap model_transformer --input_path $COLMAP_IN --output_path $COLMAP_OUT --transform_path ./transform.txt

colmap model_converter --input_path $COLMAP_OUT --output_path $COLMAP_OUT/points3D.ply --output_type PLY
```
