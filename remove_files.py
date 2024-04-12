import os
import shutil

def remove_files_with_same_name(source_folder, target_folder):
    try:
        # Get a list of all files in the source folder
        source_files = os.listdir(source_folder)
        
        # Get a list of all files in the target folder
        target_files = os.listdir(target_folder)
        
        # Loop through each file in the source folder
        for source_file in source_files:
            # Check if the file exists in the target folder
            if source_file in target_files:
                # Get the full paths of both the source and target files
                source_file_path = os.path.join(source_folder, source_file)
                target_file_path = os.path.join(target_folder, source_file)
                
                # Remove the file from the target folder
                os.remove(target_file_path)
                print(f"File '{source_file}' removed from the target folder.")
        
        print("Operation completed successfully.")
        
    except Exception as e:
        print("An error occurred:", str(e))

# Example usage:
source_folder = "/mnt/hdd/data/Okutama_Action/Yonghan_data/Data/Drone1"
target_folder = "/mnt/hdd/data/Okutama_Action/Yonghan_data/Data/images"
remove_files_with_same_name(source_folder, target_folder)
