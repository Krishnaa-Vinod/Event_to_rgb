#!/usr/bin/env python3
"""
Inspect actual H5 files and bag structure to understand the data format.
This script verifies the actual schema rather than relying on assumptions.
"""

import h5py
import os
import sys
import numpy as np
from pathlib import Path
import yaml

def inspect_h5_file(h5_path):
    """Inspect an H5 file and print its structure."""
    print(f"\n=== H5 File: {h5_path} ===")
    print(f"File size: {os.path.getsize(h5_path) / (1024**3):.2f} GB")

    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"Root keys: {list(f.keys())}")

            for key in f.keys():
                dataset = f[key]
                if hasattr(dataset, 'shape'):
                    print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")

                    # Print attributes if any
                    if dataset.attrs:
                        print(f"    attrs: {dict(dataset.attrs)}")

                    # Print a small sample for understanding
                    if key == 'timestamps_ns' and len(dataset) > 0:
                        print(f"    sample timestamps: {dataset[:5]} (first 5)")
                        print(f"    timestamp range: {dataset[0]} to {dataset[-1]}")
                    elif key == 'voxels' and len(dataset) > 0:
                        print(f"    voxel value range: [{dataset[0].min():.3f}, {dataset[0].max():.3f}]")
                    elif key == 'rgb_mask' and len(dataset) > 0:
                        print(f"    RGB mask: {dataset[:10]} (first 10 values)")
                        print(f"    RGB frames available: {dataset.sum()} out of {len(dataset)}")
                    elif key == 'rgb_images' and len(dataset) > 0:
                        print(f"    RGB image value range: [{dataset[0].min()}, {dataset[0].max()}]")
                else:
                    print(f"  {key}: {type(dataset)}")

    except Exception as e:
        print(f"Error reading H5 file: {e}")

def inspect_bag_metadata(bag_dir):
    """Inspect bag/MCAP metadata."""
    print(f"\n=== Bag Directory: {bag_dir} ===")

    bag_path = Path(bag_dir)
    if not bag_path.exists():
        print(f"Bag directory does not exist: {bag_dir}")
        return

    # List all files
    all_files = list(bag_path.iterdir())
    print(f"Files in directory: {[f.name for f in all_files]}")

    # Check metadata.yaml if it exists
    metadata_file = bag_path / "metadata.yaml"
    if metadata_file.exists():
        print("\n--- Metadata.yaml ---")
        try:
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f)
                print(yaml.dump(metadata, default_flow_style=False))
        except Exception as e:
            print(f"Error reading metadata: {e}")

    # List MCAP files and their sizes
    mcap_files = list(bag_path.glob("*.mcap"))
    for mcap_file in mcap_files:
        size_gb = mcap_file.stat().st_size / (1024**3)
        print(f"  {mcap_file.name}: {size_gb:.2f} GB")

def main():
    # Paths from the mission specification
    h5_dir = "/scratch/kvinod/bags/eGo_navi_overfit_data_h5"
    bag_dir = "/scratch/kvinod/bags/overfitting_data/data_collect_20260228_153433"

    print("=== EVENT-TO-RGB DATA INSPECTION ===")
    print(f"H5 directory: {h5_dir}")
    print(f"Bag directory: {bag_dir}")

    # Inspect bag directory
    inspect_bag_metadata(bag_dir)

    # Inspect one H5 file (the matching one)
    h5_file = os.path.join(h5_dir, "data_collect_20260228_153433.h5")
    if os.path.exists(h5_file):
        inspect_h5_file(h5_file)
    else:
        print(f"\nMatching H5 file not found: {h5_file}")
        print("Available H5 files:")
        if os.path.exists(h5_dir):
            for f in os.listdir(h5_dir):
                if f.endswith('.h5'):
                    print(f"  {f}")

    # Also inspect a second H5 file for comparison
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    if len(h5_files) > 1:
        second_h5 = os.path.join(h5_dir, h5_files[1])
        inspect_h5_file(second_h5)

    print(f"\n=== INSPECTION COMPLETE ===")

if __name__ == "__main__":
    main()