#!/usr/bin/env python3
"""
Inspect actual H5 files and bag structure to understand the data format.
This script verifies the actual schema rather than relying on assumptions.
"""

import h5py
import os
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="Inspect H5 files and bag directory structure")
    parser.add_argument("--bag-dir", required=True,
                       help="Path to bag directory to inspect")
    parser.add_argument("--h5-file", required=True,
                       help="Path to H5 file to inspect")
    parser.add_argument("--h5-dir",
                       help="Optional: H5 directory to list additional files")

    args = parser.parse_args()

    print("=== EVENT-TO-RGB DATA INSPECTION ===")
    print(f"H5 file: {args.h5_file}")
    print(f"Bag directory: {args.bag_dir}")

    # Inspect bag directory
    inspect_bag_metadata(args.bag_dir)

    # Inspect the specified H5 file
    if os.path.exists(args.h5_file):
        inspect_h5_file(args.h5_file)
    else:
        print(f"\nSpecified H5 file not found: {args.h5_file}")

    # If H5 directory provided, show other available files
    if args.h5_dir and os.path.exists(args.h5_dir):
        print(f"\nOther H5 files in {args.h5_dir}:")
        for f in os.listdir(args.h5_dir):
            if f.endswith('.h5'):
                print(f"  {f}")

    print(f"\n=== INSPECTION COMPLETE ===")

    # Schema validation
    print(f"\n=== SCHEMA VALIDATION ===")
    try:
        import h5py
        with h5py.File(args.h5_file, 'r') as f:
            has_rgb_indices = 'rgb_indices' in f
            print(f"Has rgb_indices dataset: {has_rgb_indices}")

            if 'rgb_images' in f and 'rgb_mask' in f:
                rgb_images_len = len(f['rgb_images'])
                rgb_mask_len = len(f['rgb_mask'])
                rgb_mask_sum = f['rgb_mask'][...].sum()
                print(f"RGB images length: {rgb_images_len}")
                print(f"RGB mask length: {rgb_mask_len}")
                print(f"RGB mask true count: {rgb_mask_sum}")

                if has_rgb_indices:
                    rgb_indices_len = len(f['rgb_indices'])
                    print(f"RGB indices length: {rgb_indices_len}")

                    if rgb_indices_len == rgb_images_len:
                        print("✓ RGB indices and images lengths match")
                    else:
                        print("✗ RGB indices and images lengths mismatch")

            if 'timestamps_ns' in f:
                timestamps_len = len(f['timestamps_ns'])
                print(f"Timestamps length: {timestamps_len}")

    except Exception as e:
        print(f"Error in schema validation: {e}")

if __name__ == "__main__":
    main()