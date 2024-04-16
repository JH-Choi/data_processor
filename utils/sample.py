import os
import shutil
import argparse

def construct_target_path_and_filename(original_path, src_root, tgt_root, img_name):
    """
    Constructs a target path for the image under its respective Drone directory,
    with a new filename that includes the original sub-directory paths.
    """
    relative_dir = os.path.relpath(original_path, src_root)
    parts = relative_dir.split(os.sep)
    # The first part of the path (Drone1 or Drone2) is used to construct the direct target directory
    drone_dir = parts[0]
    # The rest of the path is used in the new filename
    new_filedir = '_'.join(parts[1:])
    new_filedir = new_filedir.replace(".", ".")

    new_filename = new_filedir + "_" + img_name

    return os.path.join(tgt_root, drone_dir, new_filename), os.path.join(tgt_root, drone_dir)

def copy_and_log_images(src_root, tgt_root, interval, log_dir):
    selected_log_path = os.path.join(tgt_root, log_dir, 'selected_images.txt')
    not_selected_log_path = os.path.join(tgt_root, log_dir, 'not_selected_images.txt')
    os.makedirs(os.path.dirname(selected_log_path), exist_ok=True)

    selected_images = []
    not_selected_images = []

    for root, dirs, files in os.walk(src_root):
        if dirs and any(dir.isdigit() for dir in dirs[0].split('.')):  # Checks if current dirs are like '1.1.1'
            for dir in dirs:
                current_path = os.path.join(root, dir)

                # if "Noon" not in root:
                #     continue
                
                image_files = sorted([f for f in os.listdir(current_path) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
 

                selected = image_files[::interval]
                not_selected = [f for f in image_files if f not in selected]

                # Copy selected images and update logs
                for img in selected:
                    new_img_path, drone_dir = construct_target_path_and_filename(current_path, src_root, tgt_root, img)
                    os.makedirs(drone_dir, exist_ok=True)  # Ensure the drone directory exists
                    shutil.copy(os.path.join(current_path, img), new_img_path)
                    selected_images.append(new_img_path.replace(tgt_root + os.sep, ""))

                for img in not_selected:
                    # For logging purposes, calculate the path as if it were selected
                    not_selected_path, _ = construct_target_path_and_filename(current_path, src_root, tgt_root, img)
                    not_selected_images.append(not_selected_path.replace(tgt_root + os.sep, ""))

    # Write to log files
    with open(selected_log_path, 'w') as f:
        f.write("\n".join(selected_images))
    with open(not_selected_log_path, 'w') as f:
        f.write("\n".join(not_selected_images))

    print(f'Processing complete. Logs saved to {log_dir}.')

parser = argparse.ArgumentParser(description="Copy sampled images and log selections.")
parser.add_argument('src_root', help="Source root directory.")
parser.add_argument('tgt_root', help="Target root directory.")
parser.add_argument('--interval', '-n', type=int, default=10, help="Image sampling interval.")
parser.add_argument('--log_dir', type=str, default="logs", help="Directory under target root for saving logs.")

args = parser.parse_args()

copy_and_log_images(args.src_root, args.tgt_root, args.interval, args.log_dir)

