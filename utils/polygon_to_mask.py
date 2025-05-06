"""
This module converts the polygon labels from roboflow to binary mask.

Author: Deja S.
Version: 1.0.2
"""

import os
import json
import tqdm
import argparse
import cv2 as cv
import numpy as np
import supervision as sv
from collections import defaultdict
from charset_normalizer.md import annotations

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_dir', required=True, type=str, help="The input image directory path.")
    parser.add_argument('-l', '--label', required=True, type=str, help="The input label JSON file path.")
    parser.add_argument('-o', '--out_dir', required=True, type=str, help="The output directory path.")
    opts = parser.parse_args()

    # Required variables
    img_dir = opts.img_dir
    label_file = opts.label
    out_dir = opts.out_dir

    # If path are valid
    assert os.path.exists(img_dir)
    assert os.path.exists(label_file)

    # Create output folder
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Get all the images
    imgs = os.listdir(img_dir)
    print(f"Number of files: {len(imgs)}")

    # Read in the annotation
    with open(label_file, 'r') as file:
        data = json.load(file)

    imgs_data = data["images"]
    annotations = data["annotations"]

    # Map images to annotations
    ann_map = defaultdict(list)
    for ann in annotations:
        ann_map[ann["image_id"]].append(ann)

    # Create the binary masks
    for img in tqdm.tqdm(imgs_data):
        image_id = img["id"]
        filename = img["file_name"]
        height = img["height"]
        width = img["width"]

        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in ann_map[image_id]:
            for seg in ann["segmentation"]:
                # Reshape the flat list into (N, 2) polygon
                polygon = np.array(seg).reshape((-1, 2)).astype(np.int32)
                print(polygon)
                cv.fillPoly(mask, [polygon], color=1)

        # Save mask
        mask_filename = filename
        cv.imwrite(os.path.join(out_dir, mask_filename), mask * 255)