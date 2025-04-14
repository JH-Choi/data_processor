
##### preprocess PISR data #####
scene_name=BlackLoong
data_dir=/mnt/hdd/data/Polarization_Imaging_data/PISR_data/$scene_name/
output_folder=/mnt/hdd/data/Polarization_Imaging_data/PISR_data/$scene_name/sparse/0/

python preprocess_PISR.py --data_dir $data_dir --output_folder $output_folder



