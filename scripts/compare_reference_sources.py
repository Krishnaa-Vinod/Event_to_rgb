#!/usr/bin/env python3
"""
Compare H5 RGB references against bag-decoded RGB references.

This script validates the consistency between H5 stored references
and RGB images decoded directly from the bag file.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import h5py
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def load_h5_references(h5_file, max_frames=None):
    """Load RGB references from H5 file."""
    with h5py.File(h5_file, 'r') as f:
        rgb_images = f['rgb_images'][...]
        timestamps_ns = f['timestamps_ns'][...]

        if 'rgb_indices' in f:
            rgb_indices = f['rgb_indices'][...]
            ref_timestamps = timestamps_ns[rgb_indices]
            print(f"Using rgb_indices: {len(rgb_images)} images")
            return rgb_images[:max_frames], ref_timestamps[:max_frames]
        else:
            rgb_mask = f['rgb_mask'][...]
            valid_idx = np.where(rgb_mask)[0]
            ref_timestamps = timestamps_ns[valid_idx]
            print(f"Using rgb_mask: {len(valid_idx)} valid images")
            return rgb_images[valid_idx][:max_frames], ref_timestamps[:max_frames]

def load_bag_rgb_frames(bag_rgb_dir, max_frames=None):
    """Load RGB frames saved from bag processing."""
    bag_path = Path(bag_rgb_dir)

    if not bag_path.exists():
        print(f"Warning: Bag RGB directory not found: {bag_rgb_dir}")
        return None, None

    # Look for RGB frames
    rgb_files = list(bag_path.glob("rgb_*.png")) + list(bag_path.glob("frame_*.png"))

    if not rgb_files:
        print(f"Warning: No RGB frames found in {bag_rgb_dir}")
        return None, None

    rgb_files.sort()
    if max_frames:
        rgb_files = rgb_files[:max_frames]

    # Load images
    images = []
    timestamps = []

    for rgb_file in rgb_files:
        img = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        if img is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)

            # Extract timestamp from filename if possible
            try:
                if 'rgb_' in rgb_file.name:
                    ts_str = rgb_file.name.split('rgb_')[1].split('.')[0]
                elif 'frame_' in rgb_file.name:
                    ts_str = rgb_file.name.split('frame_')[1].split('.')[0]
                else:
                    ts_str = "0"

                timestamp = float(ts_str)
                timestamps.append(timestamp)
            except:
                # Fallback to sequential timestamps
                timestamps.append(len(timestamps) * 0.033)  # ~30 FPS

    print(f"Loaded {len(images)} RGB frames from bag")
    return np.array(images), np.array(timestamps)

def compare_reference_sources(h5_file, bag_rgb_dir, output_dir, max_frames=20):
    """Compare H5 and bag RGB references."""

    print(f"=== Reference Source Comparison ===")
    print(f"H5 file: {h5_file}")
    print(f"Bag RGB dir: {bag_rgb_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load H5 references
    try:
        h5_images, h5_timestamps = load_h5_references(h5_file, max_frames)
    except Exception as e:
        print(f"Error loading H5 references: {e}")
        return None

    # Load bag references
    bag_images, bag_timestamps = load_bag_rgb_frames(bag_rgb_dir, max_frames)

    if bag_images is None:
        print("Cannot compare - bag RGB frames not available")
        return {
            'h5_file': str(h5_file),
            'bag_rgb_dir': str(bag_rgb_dir),
            'h5_frame_count': len(h5_images),
            'bag_frame_count': 0,
            'error': 'Bag RGB frames not available',
            'comparison_possible': False
        }

    # Find temporal matches
    matches = []
    comparison_results = []

    print(f"\nTemporal matching:")
    print(f"  H5 timestamps: {h5_timestamps[0]/1e9:.3f} - {h5_timestamps[-1]/1e9:.3f}s")
    print(f"  Bag timestamps: {bag_timestamps[0]:.3f} - {bag_timestamps[-1]:.3f}s")

    # Convert H5 timestamps to seconds for matching
    h5_timestamps_s = h5_timestamps / 1e9

    for i, h5_ts in enumerate(h5_timestamps_s):
        # Find closest bag timestamp
        deltas = np.abs(bag_timestamps - h5_ts)
        closest_idx = np.argmin(deltas)
        delta_ms = deltas[closest_idx] * 1000

        if delta_ms <= 100:  # Within 100ms
            matches.append((i, closest_idx, delta_ms))

    print(f"Found {len(matches)} temporal matches within 100ms")

    if not matches:
        return {
            'h5_file': str(h5_file),
            'bag_rgb_dir': str(bag_rgb_dir),
            'h5_frame_count': len(h5_images),
            'bag_frame_count': len(bag_images),
            'temporal_matches': 0,
            'error': 'No temporal matches found',
            'comparison_possible': False
        }

    # Compare matched frames
    print(f"\nComparing {len(matches)} matched frame pairs...")

    for h5_idx, bag_idx, delta_ms in tqdm(matches, desc="Comparing frames"):
        h5_img = h5_images[h5_idx]
        bag_img = bag_images[bag_idx]

        # Resize images to same size for comparison
        target_size = (640, 480)  # Common size
        h5_resized = cv2.resize(h5_img, target_size)
        bag_resized = cv2.resize(bag_img, target_size)

        # Convert to grayscale for comparison
        h5_gray = cv2.cvtColor(h5_resized, cv2.COLOR_RGB2GRAY)
        bag_gray = cv2.cvtColor(bag_resized, cv2.COLOR_RGB2GRAY)

        # Compute metrics
        mse = float(np.mean((h5_gray.astype(float) - bag_gray.astype(float)) ** 2))
        mae = float(np.mean(np.abs(h5_gray.astype(float) - bag_gray.astype(float))))

        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = float('inf')

        ssim_val = ssim(h5_gray, bag_gray, data_range=255)

        result = {
            'h5_index': h5_idx,
            'bag_index': bag_idx,
            'timestamp_delta_ms': float(delta_ms),
            'h5_timestamp_s': float(h5_timestamps_s[h5_idx]),
            'bag_timestamp_s': float(bag_timestamps[bag_idx]),
            'mse_gray': float(mse),
            'mae_gray': float(mae),
            'psnr_gray': float(psnr),
            'ssim_gray': float(ssim_val),
            'h5_shape': h5_img.shape,
            'bag_shape': bag_img.shape
        }
        comparison_results.append(result)

    # Compute summary statistics
    if comparison_results:
        metrics = ['mse_gray', 'mae_gray', 'psnr_gray', 'ssim_gray', 'timestamp_delta_ms']
        summary_stats = {}

        for metric in metrics:
            values = [r[metric] for r in comparison_results if np.isfinite(r[metric])]
            if values:
                summary_stats[f'{metric}_mean'] = float(np.mean(values))
                summary_stats[f'{metric}_std'] = float(np.std(values))
                summary_stats[f'{metric}_median'] = float(np.median(values))
            else:
                summary_stats[f'{metric}_mean'] = 0.0
                summary_stats[f'{metric}_std'] = 0.0
                summary_stats[f'{metric}_median'] = 0.0

    # Determine alignment quality
    if len(comparison_results) > 0:
        avg_psnr = summary_stats['psnr_gray_mean']
        avg_ssim = summary_stats['ssim_gray_mean']
        avg_delta = summary_stats['timestamp_delta_ms_mean']

        if avg_psnr > 25 and avg_ssim > 0.8:
            alignment_quality = "EXCELLENT"
        elif avg_psnr > 20 and avg_ssim > 0.7:
            alignment_quality = "GOOD"
        elif avg_psnr > 15 and avg_ssim > 0.5:
            alignment_quality = "FAIR"
        else:
            alignment_quality = "POOR"
    else:
        alignment_quality = "UNKNOWN"

    # Create final results
    results = {
        'h5_file': str(h5_file),
        'bag_rgb_dir': str(bag_rgb_dir),
        'h5_frame_count': len(h5_images),
        'bag_frame_count': len(bag_images),
        'temporal_matches': len(matches),
        'valid_comparisons': len(comparison_results),
        'comparison_possible': len(comparison_results) > 0,
        'alignment_quality': alignment_quality,
        'summary_statistics': summary_stats if comparison_results else {},
        'per_frame_comparisons': comparison_results
    }

    # Save results
    results_file = output_path / "reference_source_consistency.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Create CSV summary
    if comparison_results:
        df = pd.DataFrame(comparison_results)
        csv_file = output_path / "reference_source_consistency.csv"
        df.to_csv(csv_file, index=False)

    # Print summary
    print(f"\n✓ Reference source comparison complete:")
    print(f"  H5 frames: {len(h5_images)}")
    print(f"  Bag frames: {len(bag_images) if bag_images is not None else 0}")
    print(f"  Valid matches: {len(comparison_results)}")
    print(f"  Alignment quality: {alignment_quality}")
    if comparison_results:
        print(f"  Avg PSNR: {summary_stats['psnr_gray_mean']:.1f} dB")
        print(f"  Avg SSIM: {summary_stats['ssim_gray_mean']:.3f}")
        print(f"  Avg time delta: {summary_stats['timestamp_delta_ms_mean']:.1f} ms")
    print(f"  Results saved: {results_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Compare H5 and bag RGB references")
    parser.add_argument("--h5-file", required=True, help="H5 file with RGB references")
    parser.add_argument("--bag-rgb-dir", required=True, help="Directory with bag RGB frames")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--max-frames", type=int, default=20, help="Max frames to compare")

    args = parser.parse_args()

    if not os.path.exists(args.h5_file):
        print(f"Error: H5 file not found: {args.h5_file}")
        sys.exit(1)

    results = compare_reference_sources(
        h5_file=args.h5_file,
        bag_rgb_dir=args.bag_rgb_dir,
        output_dir=args.output,
        max_frames=args.max_frames
    )

    if results is None or not results['comparison_possible']:
        print("⚠️ Reference comparison failed or not possible")
        sys.exit(1)
    else:
        print(f"✓ Reference comparison successful: {results['alignment_quality']}")

if __name__ == "__main__":
    main()