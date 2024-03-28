
# How to use (usage examples)
```
$ python convert_to_colmap.py --data_dir ${DATA_PATH} --resolution 4
$ python process_colmap_sfm.py --colmap_dir ${DATA_PATH}/colmap_data --image_path ${DATA_PATH}
$ python split_colmap_model.py --input_path ${DATA_PATH}/colmap_data --output_path ${DATA_PATH}/colmap_aligned --tiles "4,2,1" --transform_path ${CODE_PATH}/configs/Mill19/transform.txt --bbox_path ${CODE_PATH}/configs/Mill19/bbox.txt
```

# Useful utils

To generate the transform.txt for other dataset, you can use get_transform_from_ref_images.py
```
$ python utils/get_transform_from_ref_images.py --input_path ${DATA_PATH}/colmap_data --out_transform_path transform.txt --idx_o=941 --ids_x "941, 1034", --ids_y "1939, 1915"
```
