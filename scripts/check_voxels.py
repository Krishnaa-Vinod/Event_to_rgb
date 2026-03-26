#!/usr/bin/env python3
"""
Quick script to get the exact voxel dimensions from H5 files.
"""
import h5py
import numpy as np
import sys

def inspect_voxels(h5_path):
    print(f"=== Inspecting voxels in {h5_path} ===")
    with h5py.File(h5_path, 'r') as f:
        voxels = f['voxels']
        timestamps = f['timestamps_ns']
        rgb_mask = f['rgb_mask']

        print(f"Voxels shape: {voxels.shape}")
        print(f"Voxels dtype: {voxels.dtype}")
        print(f"Timestamps shape: {timestamps.shape}")

        # Count valid RGB frames
        rgb_count = np.sum(rgb_mask[:])
        total_frames = len(rgb_mask)
        print(f"RGB frames: {rgb_count} out of {total_frames} total")

        # Sample voxel statistics
        first_voxel = voxels[0]
        print(f"First voxel range: [{first_voxel.min():.4f}, {first_voxel.max():.4f}]")

        # Check timestamp intervals
        if len(timestamps) > 1:
            dt = np.diff(timestamps[:10]) / 1e6  # Convert to ms
            print(f"First 9 time deltas (ms): {dt}")
            print(f"Mean time delta (ms): {np.mean(dt):.2f}")

        return voxels.shape

if __name__ == "__main__":
    h5_file = "/scratch/kvinod/bags/eGo_navi_overfit_data_h5/data_collect_20260228_153433.h5"
    voxel_shape = inspect_voxels(h5_file)