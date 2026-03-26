#!/usr/bin/env python3
"""
Evaluation metrics pipeline for event-to-RGB reconstruction.

This script computes quality metrics between reconstructed images and
ground truth RGB references, supporting multiple reconstruction methods.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from skimage.metrics import structural_similarity as ssim

def load_rgb_references(h5_file):
    """Load RGB reference images from H5 file."""
    try:
        import h5py
        with h5py.File(h5_file, 'r') as f:
            rgb_images = f['rgb_images'][...]  # (N, 1024, 1280, 3)
            timestamps_ns = f['timestamps_ns'][...]
            rgb_mask = f['rgb_mask'][...]

        # Filter to only frames with RGB data
        valid_indices = np.where(rgb_mask)[0]
        rgb_images = rgb_images[valid_indices]
        timestamps_ns = timestamps_ns[valid_indices]

        print(f"Loaded {len(rgb_images)} RGB reference images")
        return rgb_images, timestamps_ns, valid_indices

    except Exception as e:
        print(f"Error loading RGB references: {e}")
        return None, None, None

def preprocess_image_for_comparison(img, target_shape=(720, 1280), to_grayscale=True):
    """
    Preprocess image for fair comparison.

    Args:
        img: Input image (grayscale or color)
        target_shape: Target height, width
        to_grayscale: Convert to grayscale for comparison
    """

    # Handle different input formats
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Color image
        if to_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        # Single channel with 3D shape
        img = img.squeeze()

    # Resize to target shape if needed
    if img.shape[:2] != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_LINEAR)

    # Ensure uint8 format
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img

def compute_image_metrics(reconstruction, reference):
    """
    Compute quality metrics between reconstruction and reference.

    Args:
        reconstruction: Reconstructed image (grayscale, uint8)
        reference: Reference image (grayscale, uint8)

    Returns:
        Dictionary of metrics
    """

    # Ensure images have same shape and type
    assert reconstruction.shape == reference.shape, f"Shape mismatch: {reconstruction.shape} vs {reference.shape}"
    assert reconstruction.dtype == reference.dtype == np.uint8

    metrics = {}

    # Mean Squared Error (MSE)
    mse = np.mean((reconstruction.astype(float) - reference.astype(float)) ** 2)
    metrics['mse'] = float(mse)

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(reconstruction.astype(float) - reference.astype(float)))
    metrics['mae'] = float(mae)

    # Peak Signal-to-Noise Ratio (PSNR)
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    metrics['psnr'] = float(psnr)

    # Structural Similarity Index (SSIM)
    try:
        ssim_value = ssim(reference, reconstruction, data_range=255)
        metrics['ssim'] = float(ssim_value)
    except Exception as e:
        print(f"Warning: SSIM computation failed: {e}")
        metrics['ssim'] = 0.0

    # Additional basic metrics
    metrics['reconstruction_mean'] = float(np.mean(reconstruction))
    metrics['reference_mean'] = float(np.mean(reference))
    metrics['reconstruction_std'] = float(np.std(reconstruction))
    metrics['reference_std'] = float(np.std(reference))

    return metrics

def find_matching_frames(reconstruction_timestamps, reference_timestamps, max_delta_ms=50):
    """
    Find temporal matches between reconstruction and reference frames.

    Args:
        reconstruction_timestamps: List of reconstruction timestamps (seconds)
        reference_timestamps: List of reference timestamps (nanoseconds)
        max_delta_ms: Maximum allowed time difference in milliseconds

    Returns:
        List of (recon_idx, ref_idx, delta_ms) tuples
    """

    matches = []
    ref_timestamps_s = reference_timestamps / 1e9  # Convert to seconds

    for recon_idx, recon_ts in enumerate(reconstruction_timestamps):
        # Find closest reference timestamp
        deltas = np.abs(ref_timestamps_s - recon_ts)
        closest_ref_idx = np.argmin(deltas)
        delta_ms = deltas[closest_ref_idx] * 1000

        if delta_ms <= max_delta_ms:
            matches.append((recon_idx, closest_ref_idx, delta_ms))

    print(f"Found {len(matches)} temporal matches within {max_delta_ms}ms")
    return matches

def evaluate_reconstruction_method(
    reconstruction_dir,
    h5_reference_file,
    method_name,
    output_dir,
    max_delta_ms=50
):
    """
    Evaluate a single reconstruction method against H5 references.

    Args:
        reconstruction_dir: Directory containing reconstructed images
        h5_reference_file: H5 file with RGB references
        method_name: Name of the reconstruction method
        output_dir: Output directory for results

    Returns:
        Evaluation summary dictionary
    """

    print(f"\n=== Evaluating {method_name.upper()} ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load RGB references
    rgb_refs, ref_timestamps, ref_indices = load_rgb_references(h5_reference_file)
    if rgb_refs is None:
        return None

    # Find reconstruction images
    recon_path = Path(reconstruction_dir)
    recon_files = list(recon_path.glob("*.png"))

    if not recon_files:
        print(f"No PNG files found in {reconstruction_dir}")
        return None

    recon_files.sort()
    print(f"Found {len(recon_files)} reconstruction images")

    # Load reconstruction metadata if available
    metadata_files = list(recon_path.glob("*summary.json"))
    recon_timestamps = []

    if metadata_files:
        # Try to load timestamps from metadata
        try:
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)

            if 'frames' in metadata:
                for frame_info in metadata['frames']:
                    recon_timestamps.append(frame_info.get('timestamp', 0))
            elif 'reconstructions' in metadata:
                for frame_info in metadata['reconstructions']:
                    recon_timestamps.append(frame_info.get('timestamp_ns', 0) / 1e9)

            print(f"Loaded {len(recon_timestamps)} timestamps from metadata")

        except Exception as e:
            print(f"Could not load metadata: {e}")

    # If no timestamps available, create dummy ones
    if len(recon_timestamps) != len(recon_files):
        print("Using dummy timestamps for reconstructions")
        # Assume 250ms intervals starting from reference start time
        start_time = ref_timestamps[0] / 1e9
        recon_timestamps = [start_time + i * 0.25 for i in range(len(recon_files))]

    # Find temporal matches
    matches = find_matching_frames(recon_timestamps, ref_timestamps, max_delta_ms)

    if not matches:
        print(f"No temporal matches found for {method_name}")
        return None

    # Compute metrics for matched frames
    frame_metrics = []
    valid_comparisons = 0

    print(f"Computing metrics for {len(matches)} matched frames...")

    for recon_idx, ref_idx, delta_ms in tqdm(matches, desc="Processing matches"):

        try:
            # Load reconstruction image
            recon_file = recon_files[recon_idx]
            recon_img = cv2.imread(str(recon_file), cv2.IMREAD_GRAYSCALE)

            if recon_img is None:
                print(f"Could not load {recon_file}")
                continue

            # Get reference image
            ref_img = rgb_refs[ref_idx]  # (1024, 1280, 3)

            # Preprocess for comparison
            recon_processed = preprocess_image_for_comparison(
                recon_img, target_shape=(720, 1280), to_grayscale=True
            )

            ref_processed = preprocess_image_for_comparison(
                ref_img, target_shape=(720, 1280), to_grayscale=True
            )

            # Compute metrics
            metrics = compute_image_metrics(recon_processed, ref_processed)

            # Add frame-specific info
            frame_result = {
                'recon_index': recon_idx,
                'ref_index': ref_idx,
                'recon_file': recon_file.name,
                'timestamp_delta_ms': float(delta_ms),
                'recon_timestamp_s': float(recon_timestamps[recon_idx]),
                'ref_timestamp_ns': int(ref_timestamps[ref_idx]),
                **metrics
            }

            frame_metrics.append(frame_result)
            valid_comparisons += 1

        except Exception as e:
            print(f"Error processing frame {recon_idx}: {e}")
            continue

    if not frame_metrics:
        print(f"No valid comparisons for {method_name}")
        return None

    # Compute aggregate metrics
    metric_names = ['mse', 'mae', 'psnr', 'ssim']
    aggregate_metrics = {}

    for metric in metric_names:
        values = [f[metric] for f in frame_metrics if metric in f and np.isfinite(f[metric])]
        if values:
            aggregate_metrics[f'{metric}_mean'] = float(np.mean(values))
            aggregate_metrics[f'{metric}_std'] = float(np.std(values))
            aggregate_metrics[f'{metric}_median'] = float(np.median(values))
        else:
            aggregate_metrics[f'{metric}_mean'] = 0.0
            aggregate_metrics[f'{metric}_std'] = 0.0
            aggregate_metrics[f'{metric}_median'] = 0.0

    # Time delta statistics
    deltas = [f['timestamp_delta_ms'] for f in frame_metrics]
    aggregate_metrics['mean_timestamp_delta_ms'] = float(np.mean(deltas))
    aggregate_metrics['median_timestamp_delta_ms'] = float(np.median(deltas))

    # Create evaluation summary
    summary = {
        'method': method_name,
        'reconstruction_dir': str(reconstruction_dir),
        'reference_file': str(h5_reference_file),
        'total_reconstructions': len(recon_files),
        'total_references': len(rgb_refs),
        'valid_matches': len(matches),
        'valid_comparisons': valid_comparisons,
        'max_delta_ms': max_delta_ms,
        'aggregate_metrics': aggregate_metrics,
        'per_frame_metrics': frame_metrics
    }

    # Save results
    results_file = output_path / f"{method_name}_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save CSV for easy analysis
    if frame_metrics:
        df = pd.DataFrame(frame_metrics)
        csv_file = output_path / f"{method_name}_metrics.csv"
        df.to_csv(csv_file, index=False)

    print(f"\n✓ {method_name.upper()} Evaluation Results:")
    print(f"  Valid comparisons: {valid_comparisons}")
    print(f"  Mean PSNR: {aggregate_metrics.get('psnr_mean', 0):.2f} dB")
    print(f"  Mean SSIM: {aggregate_metrics.get('ssim_mean', 0):.3f}")
    print(f"  Mean timestamp delta: {aggregate_metrics.get('mean_timestamp_delta_ms', 0):.2f} ms")
    print(f"  Results saved: {results_file}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction methods")
    parser.add_argument("--reconstruction-dir", required=True,
                       help="Directory containing reconstructed images")
    parser.add_argument("--reference-h5", required=True,
                       help="H5 file with RGB references")
    parser.add_argument("--method-name", required=True,
                       help="Name of reconstruction method")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for evaluation results")
    parser.add_argument("--max-delta", type=float, default=50,
                       help="Max timestamp delta in milliseconds")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.reconstruction_dir):
        print(f"Error: Reconstruction directory not found: {args.reconstruction_dir}")
        sys.exit(1)

    if not os.path.exists(args.reference_h5):
        print(f"Error: Reference H5 file not found: {args.reference_h5}")
        sys.exit(1)

    # Run evaluation
    summary = evaluate_reconstruction_method(
        reconstruction_dir=args.reconstruction_dir,
        h5_reference_file=args.reference_h5,
        method_name=args.method_name,
        output_dir=args.output,
        max_delta_ms=args.max_delta
    )

    if summary:
        print(f"\n✓ Evaluation complete for {args.method_name}")
        print(f"Valid comparisons: {summary['valid_comparisons']}")
    else:
        print(f"\n✗ Evaluation failed for {args.method_name}")
        sys.exit(1)

if __name__ == "__main__":
    main()