"""
Preprocessing Script for Palm Oil Fresh Fruit Bunch (FFB) Dataset
Supports IPB 3-class directory structure (unripe/, ripe/, overripe/) or flat JPG/PNG files.
Converts raw RGB images into 32x32 RGB BMP & signed INT8 HEX files for Cyclone V FPGA verification.
"""

import os
import glob
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "sample_raw")
OUT_DIR = os.path.join(BASE_DIR, "processed_32x32")

def preprocess_dataset(target_size=(32, 32)):
    image_paths = sorted(
        glob.glob(os.path.join(RAW_DIR, "**", "*.jpg"), recursive=True) +
        glob.glob(os.path.join(RAW_DIR, "**", "*.png"), recursive=True)
    )
    if not image_paths:
        print("No JPG/PNG images found in sample_raw/")
        return

    count = 0
    for img_path in image_paths:
        rel_dir = os.path.relpath(os.path.dirname(img_path), RAW_DIR)
        target_folder = os.path.join(OUT_DIR, rel_dir) if rel_dir != "." else OUT_DIR
        os.makedirs(target_folder, exist_ok=True)

        fname = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(target_size, Image.Resampling.BILINEAR)

        # Save 32x32 RGB BMP
        bmp_path = os.path.join(target_folder, f"{fname}.bmp")
        img_resized.save(bmp_path, format="BMP")

        # Convert [0, 255] uint8 RGB to [-128, 127] signed INT8
        arr_uint8 = np.array(img_resized, dtype=np.uint8)
        arr_int8 = (arr_uint8.astype(np.int16) - 128).astype(np.int8)

        hex_path = os.path.join(target_folder, f"{fname}_rgb.hex")
        with open(hex_path, "w") as f:
            for val in arr_int8.flatten():
                f.write(f"{int(val) & 0xFF:02x}\n")
        count += 1

    print(f"Successfully preprocessed {count} RGB images into 32x32 BMP & INT8 HEX.")

if __name__ == "__main__":
    preprocess_dataset()
