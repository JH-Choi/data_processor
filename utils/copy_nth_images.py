import shutil
import os
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Copy every nth image from source to target directory.')
parser.add_argument('src_dir', type=str, help='Source directory containing the images')
parser.add_argument('tgt_dir', type=str, help='Target directory to copy the images to')
parser.add_argument('-n', '--interval', type=int, default=10, help='Interval for selecting images to copy (e.g., every nth image)')

# Parse arguments
args = parser.parse_args()

source_directory = args.src_dir
destination_directory = args.tgt_dir
n = args.interval

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# List all files in the source directory and sort them to ensure correct order
files = sorted([f for f in os.listdir(source_directory) if f.endswith('.jpg') and f.split('.')[0].isdigit()], key=lambda x: int(x.split('.')[0]))

# Copy every nth file
for i, file in enumerate(files):
    if i % n == 0:  # This checks if the file is every nth file
        source_path = os.path.join(source_directory, file)
        destination_path = os.path.join(destination_directory, file)
        shutil.copy(source_path, destination_path)
        print(f'Copied {file} to {destination_directory}')
