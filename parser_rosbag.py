#!/usr/bin/env python3
"""
ROS bag parser: Extract compressed RGB images from a specified topic and save as PNG files
with timestamp-based filenames and generate an index file (rgb.txt).
Usage:
    python bag2png_parser.py <bag_file> -o <output_dir> [-t <topic>] [--start <start_time>] [--end <end_time>]
"""
import os
import argparse
from pathlib import Path

import rosbag
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract compressed images from ROS bag and save as PNGs with timestamps"
    )
    parser.add_argument(
        "bag_file",
        type=Path,
        help="Path to the input ROS bag file (e.g., data.bag)",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=Path("output"),
        help="Base directory to save 'rgb' folder and 'rgb.txt'",
    )
    parser.add_argument(
        "-t", "--topic",
        type=str,
        default="zed_node/rgb/image_rect_color/compressed",
        help="ROS topic to read compressed images from",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds (relative to bag start) to begin extraction",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds (relative to bag start) to stop extraction",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bag_path = args.bag_file
    topic = args.topic
    out_base = args.output_dir

    # Prepare output directories and index file
    rgb_dir = out_base / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_base / "rgb.txt"

    # Open bag
    bag = rosbag.Bag(str(bag_path), "r")
    start_ts = None
    if args.start is not None:
        start_ts = bag.get_start_time() + args.start
    end_ts = None
    if args.end is not None:
        end_ts = bag.get_start_time() + args.end

    count = 0
    with open(index_file, 'w') as idx:
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            ts = t.to_sec()
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                break

            # Decode image from compressed data
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Warning: failed to decode image at time {ts}")
                continue

            # Format timestamp filename
            ts_str = f"{ts:.6f}"
            filename = f"{ts_str}.png"
            save_path = rgb_dir / filename

            # Write image
            cv2.imwrite(str(save_path), image)
            print(f"Saved {save_path}")

            # Write to index file (relative path)
            idx.write(f"{ts_str} rgb/{filename}\n")
            count += 1

    bag.close()
    print(f"Extraction complete: {count} images saved to '{rgb_dir}', index at '{index_file}'")


if __name__ == "__main__":
    main()

