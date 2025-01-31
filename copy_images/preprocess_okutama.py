import os
import cv2
import numpy as np 
from pathlib import Path
from tqdm import tqdm

import imageio
import cv2

def variance_of_laplacian(image: np.ndarray):
    # Compute the variance of the Laplacian which measure the focus.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


train_drone1_morning_scenes = ['1.1.1', '1.1.2', '1.1.3', '1.1.4', '1.1.5', '1.1.6', '1.1.7', '1.1.10', '1.1.11'] 
train_drone1_noon_scenes = ['1.2.2', '1.2.4', '1.2.5', '1.2.6', '1.2.7', '1.2.8', '1.2.9', '1.2.11']
train_drone2_morning_scenes = ['2.1.1', '2.1.2', '2.1.3', '2.1.4', '2.1.5', '2.1.6', '2.1.7', '2.1.10']
train_drone2_noon_scenes = ['2.2.2', '2.2.4', '2.2.5', '2.2.6', '2.2.7', '2.2.8', '2.2.9', '2.2.11']
val_drone1_morning_scenes = ['1.1.8', '1.1.9']
val_drone1_noon_scenes = ['1.2.1', '1.2.3', '1.2.10']
val_drone2_morning_scenes = ['2.1.8', '2.1.9']
val_drone2_noon_scenes = ['2.2.1', '2.2.3', '2.2.10']   

########################
# Scenenario1.1.1 : 1.1.1 / 50-2270 
# Scenenario2.1.1 : 2.1.1 / 80-1252 
########################
root_path = '/mnt/hdd/data/Okutama_Action/TrainSetFrames'
out_path = '/mnt/hdd/data/Okutama_Action/GS_data/Scenario3'
# input_scenes = ['1.2.2', '2.2.2', '1.2.4', '2.2.4', '1.2.6', '1.2.8', '2.2.8', '1.2.9', '2.2.9', '1.2.11', '2.2.11']
input_scenes = ['1.1.1', '2.1.1', '1.1.2', '2.1.2', '1.1.3', '2.1.3', '1.1.4', '2.1.4', '1.1.5', '2.1.5', '1.1.6', '2.1.6', '1.1.7', '2.1.7', '1.1.10', '2.1.10', '1.1.11']
sampling_idx = 15
output_folder = os.path.join(out_path, 'images')
output_mask_folder = os.path.join(out_path, 'masks')
output_bbox_folder = os.path.join(out_path, 'bbox')
labels_folder = "/mnt/hdd/data/Okutama_Action/TrainSetFrames/Labels/MultiActionLabels/3840x2160"
create_mask = True
create_bbox = True
use_clean_frames = False # Remove the frames that have noisy bounding boxes

remove_blur_images= False
blur_filter_perc = 95.0

start_end_scenes = {
    '1.1.1': (50, 2272),
    '1.1.3': (101, 1965),
    '1.1.5': (120, 1559),
    '1.1.6': (462, 2145),
    '1.1.10': (420, 1601),
    '2.1.1': (80, 1252),
    '2.1.2': (180, 1397),
    '2.1.3': (10, 2877),
    '2.1.4': (100, 2107),
    '2.1.4': (100, 2107),
    '1.2.11': (0, 1583),
    '2.2.2': (150, 1465), 
    '2.2.11': (0, 776)
}

## Frames includes highly accurate bounding boxes
clean_frame_idxs = {
    '1.1.1': [i for i in range(50, 531)] + [i for i in range(600, 891)],
}


WIDTH, HEIGHT = 3840, 2160
resizsed_width, resizsed_height = 1280, 720
scale_x, scale_y = resizsed_width / WIDTH, resizsed_height / HEIGHT


root_path = Path(root_path)
output_folder = Path(output_folder)
output_folder.mkdir(exist_ok=True, parents=True)
output_mask_folder = Path(output_mask_folder)
output_mask_folder.mkdir(exist_ok=True, parents=True)
output_bbox_folder = Path(output_bbox_folder)
output_bbox_folder.mkdir(exist_ok=True, parents=True)

# Read the labels
bboxes_dict = {}
people_dict = {}
same_people = {}

if remove_blur_images:
    blur_scores = []
    blur_output_paths = []
    blur_mask_paths = []
    blur_bbox_paths = []

for txt_name in input_scenes:
    bboxes_dict[txt_name] = {}
    people_dict[txt_name] = set()
    same_people[txt_name] = {}


    txt_file = os.path.join(labels_folder, f'{txt_name}.txt')
    with open(txt_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        s = line.split(" ")
        frame = int(s[5])
        if frame not in bboxes_dict[txt_name]:
            bboxes_dict[txt_name][frame] = set()
        new_coord = (int(s[1]),int(s[2]),int(s[3]),int(s[4]),int(s[0]))
        # xmin, ymin, xmax, ymax, _, label
        curr_person = int(s[0])
        if curr_person not in same_people[txt_name]:
            same_people[txt_name][curr_person] = set()
        
        xc = (new_coord[0] + new_coord[2]) / 2
        yc = (new_coord[1] + new_coord[3]) / 2
        if len(bboxes_dict[txt_name][frame]) == 0:
            bboxes_dict[txt_name][frame].add(new_coord)
        else:
            add_bbox = True
            for (x1,y1,x2,y2,t) in bboxes_dict[txt_name][frame]:
                cxc = (x1 + x2) / 2
                cyc = (y1 + y2) / 2
                dist = np.sqrt((xc - cxc)**2 + (yc - cyc)**2)
                if dist < 20:
                    add_bbox = False
                    same_people[txt_name][curr_person].add(t)
            if add_bbox:
                bboxes_dict[txt_name][frame].add(new_coord)

num_image = 0 
for scene_split in input_scenes:
    if scene_split in train_drone1_morning_scenes:
        drone, time = 'Drone1', 'Morning'
    elif scene_split in train_drone1_noon_scenes:
        drone, time = 'Drone1', 'Noon'
    elif scene_split in train_drone2_morning_scenes:
        drone, time = 'Drone2', 'Morning'
    elif scene_split in train_drone2_noon_scenes:  
        drone, time = 'Drone2', 'Noon'
    else:
        raise ValueError(f'Invalid scene split: {scene_split}')

    if scene_split in start_end_scenes.keys():
        start_idx, end_idx = start_end_scenes[scene_split]
        cut_start_end_frames = True
    else:
        cut_start_end_frames = False

    source_folder = root_path / drone / time / 'Extracted-Frames-1280x720' / scene_split

    image_files = [file for file in source_folder.iterdir() if file.is_file() and (file.suffix.lower() in ['.jpg', '.png', '.jpeg'])]
    sorted_image_files = sorted(image_files, key=lambda x: int(x.name[:-4]))
    for idx, image_file in enumerate(sorted_image_files):
        if cut_start_end_frames:
            if idx < start_idx or idx > end_idx:
                continue # skip the frames that are not in the range
    
        if use_clean_frames:
            if idx not in clean_frame_idxs[scene_split]:
                continue # skip the frames that have noisy bounding boxes

        if idx % sampling_idx == 0:
            scene_split_name = scene_split.replace('.', '_')
            output_file = output_folder / f'{drone}_{time}_{scene_split_name}_{image_file.name}'
            os.system(f'cp {image_file} {output_file}')
            num_image += 1

            if create_mask or create_bbox:
                mask_file = output_mask_folder / f'{drone}_{time}_{scene_split_name}_{image_file.name}'
                bbox_file = output_bbox_folder / f'{drone}_{time}_{scene_split_name}_{image_file.name}'

                if int(image_file.name[:-4]) not in bboxes_dict[scene_split]:
                    continue
                
                label_ = bboxes_dict[scene_split][int(image_file.name[:-4])]

                if create_mask:
                    raw_img = np.zeros((resizsed_height, resizsed_width, 3), dtype=np.uint8)  # White image

                if create_bbox:
                    bbox_img = cv2.imread(str(image_file))

                for (xmin, ymin, xmax, ymax, label) in label_:
                    start_coord = (int(xmin*scale_x), int(ymin*scale_y)) # xmin, ymin
                    end_coord = (int(xmax*scale_x), int(ymax*scale_y)) # xmax, ymax
                    x_c = (xmin + ((xmax - xmin) / 2)) / WIDTH
                    y_c = (ymin + ((ymax - ymin) / 2)) / HEIGHT
                    w = (xmax - xmin) / WIDTH
                    h = (ymax - ymin) / HEIGHT

                    if create_mask:
                        raw_img = cv2.rectangle(raw_img,start_coord,end_coord,(255,255,255),-1)
                    if create_bbox:
                        bbox_img = cv2.rectangle(bbox_img, start_coord, end_coord, (0, 0, 255), 5)

                if create_mask:
                    cv2.imwrite(str(mask_file), raw_img)
                if create_bbox:
                    cv2.imwrite(str(bbox_file), bbox_img)
            

            if remove_blur_images:
                img_ = imageio.imread(str(output_file)) 
                blur_scores.append(variance_of_laplacian(img_))
                blur_output_paths.append(output_file)
                if create_mask:
                    blur_mask_paths.append(str(mask_file))
                if create_bbox:
                    blur_bbox_paths.append(str(bbox_file))

print(f'Number of images: {num_image}')

if remove_blur_images:
    blur_scores = np.array(blur_scores)
    blur_thres = np.percentile(blur_scores, blur_filter_perc)
    blur_filter_inds = np.where(blur_scores >= blur_thres)[0]
    for i in blur_filter_inds:
        print(f'Removed {blur_output_paths[i]}')
        print(f'Removed {blur_mask_paths[i]}')
        print(f'Removed {blur_bbox_paths[i]}')

