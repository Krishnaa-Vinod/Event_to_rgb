#!/usr/bin/env python3
"""
H5 direct reconstruction using existing E2VID infrastructure.

This script:
1. Extracts voxel tensors from H5 files
2. Converts them to the event format expected by E2VID
3. Runs E2VID reconstruction using their run_reconstruction.py script
"""

import os
import sys
import argparse
import h5py
import numpy as np
import json
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

def h5_to_event_format(h5_file, output_event_file, max_frames=None):
    """
    Convert H5 voxel data back to event format for E2VID processing.

    Note: This is an approximation since we're converting voxels back to events.
    The H5 direct route may not be exactly equivalent to bag->E2VID route.
    """

    print(f"Converting H5 voxels to event format: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        voxels = f['voxels'][...]  # (N, 5, 720, 1280)
        timestamps_ns = f['timestamps_ns'][...]

        print(f"Loaded {len(voxels)} voxel frames")
        print(f"Voxel shape: {voxels.shape}")

        if max_frames and len(voxels) > max_frames:
            voxels = voxels[:max_frames]
            timestamps_ns = timestamps_ns[:max_frames]

    height, width = voxels.shape[-2:]

    # Write E2VID format: first line is width height
    with open(output_event_file, 'w') as f:
        f.write(f"{width} {height}\n")

        # For each voxel frame, create pseudo-events
        # This is an approximation - we're creating events from voxel data
        event_count = 0

        for frame_idx, (voxel, timestamp_ns) in enumerate(tqdm(zip(voxels, timestamps_ns), desc="Converting voxels")):

            # Convert timestamp to seconds (E2VID expects seconds)
            timestamp_s = timestamp_ns / 1e9

            # For each temporal bin in the voxel
            for bin_idx in range(voxel.shape[0]):  # 5 temporal bins

                bin_data = voxel[bin_idx]  # (720, 1280)

                # Create time within this bin
                bin_time_s = timestamp_s + (bin_idx - 2) * 0.25 / 5  # Spread bins over 250ms window

                # Find pixels with significant activity
                # Use a threshold to create sparse events
                pos_mask = bin_data > 0.1  # Positive events
                neg_mask = bin_data < -0.1  # Negative events

                # Get coordinates for positive events
                pos_y, pos_x = np.where(pos_mask)
                for y, x in zip(pos_y, pos_x):
                    f.write(f"{bin_time_s:.9f} {x} {y} 1\n")
                    event_count += 1

                # Get coordinates for negative events
                neg_y, neg_x = np.where(neg_mask)
                for y, x in zip(neg_y, neg_x):
                    f.write(f"{bin_time_s:.9f} {x} {y} 0\n")
                    event_count += 1

    print(f"Generated {event_count} pseudo-events from H5 voxels")
    return event_count

def run_e2vid_reconstruction(event_file, weights_path, output_dir, window_duration_ms=250):
    """Run E2VID reconstruction using their official script."""

    e2vid_script = "third_party/rpg_e2vid/run_reconstruction.py"

    if not os.path.exists(e2vid_script):
        print(f"E2VID script not found: {e2vid_script}")
        return False

    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return False

    # Prepare E2VID command
    cmd = [
        "python", e2vid_script,
        "--path_to_model", weights_path,
        "--input_file", event_file,
        "--fixed_duration",
        "--window_duration", str(window_duration_ms),  # E2VID expects milliseconds
        "--output_folder", output_dir,
        "--dataset_name", "h5_reconstruction",
        "--auto_hdr"
    ]

    print(f"Running E2VID: {' '.join(cmd)}")

    try:
        # Run E2VID reconstruction
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            print("✓ E2VID reconstruction completed successfully")
            return True
        else:
            print(f"✗ E2VID reconstruction failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("✗ E2VID reconstruction timed out")
        return False
    except Exception as e:
        print(f"✗ Error running E2VID: {e}")
        return False

def run_h5_reconstruction(h5_file, output_dir, model_name='e2vid', max_frames=None, window_duration_ms=250):
    """
    Main function to run H5 reconstruction.

    Args:
        h5_file: Path to H5 file
        output_dir: Output directory
        model_name: 'e2vid' or 'firenet'
        max_frames: Max frames to process
        window_duration_ms: Window duration for reconstruction
    """

    print(f"=== H5 Direct Reconstruction: {model_name.upper()} (APPROXIMATE) ===")
    print("WARNING: H5-direct uses voxel->pseudo-event conversion and is approximate!")
    print(f"H5 file: {h5_file}")
    print(f"Output: {output_dir}")
    print(f"Max frames: {max_frames}")
    print(f"Window duration: {window_duration_ms}ms")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set weights path
    if model_name == 'e2vid':
        weights_path = "weights/E2VID_lightweight.pth.tar"
    elif model_name == 'firenet':
        weights_path = "weights/firenet.pth.tar"
        print("Warning: FireNet support is experimental")
    else:
        print(f"Unknown model: {model_name}")
        return False

    if not os.path.exists(weights_path):
        print(f"Model weights not found: {weights_path}")
        print("Run ./scripts/fetch_models.sh to download weights")
        return False

    # Create temporary event file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        temp_event_file = tmp_file.name

    try:
        # Convert H5 to event format
        print("\n1. Converting H5 voxels to event format...")
        event_count = h5_to_event_format(h5_file, temp_event_file, max_frames)

        if event_count == 0:
            print("No events generated from H5 file")
            return False

        # Run reconstruction
        print(f"\n2. Running {model_name.upper()} reconstruction...")
        success = run_e2vid_reconstruction(
            temp_event_file,
            weights_path,
            str(output_path),
            window_duration_ms
        )

        if success:
            print(f"\n✓ Reconstruction complete! Check {output_path}")

            # Create per-frame manifest by reading the E2VID output
            manifest_frames = []
            e2vid_output_imgs = list(output_path.glob("**/*.png"))
            e2vid_output_imgs.sort()

            print(f"Found {len(e2vid_output_imgs)} output frames for manifest")

            # Match output frames to original timestamps
            with h5py.File(h5_file, 'r') as f:
                if max_frames and len(f['timestamps_ns']) > max_frames:
                    h5_timestamps = f['timestamps_ns'][:max_frames]
                else:
                    h5_timestamps = f['timestamps_ns'][...]

            for i, img_file in enumerate(e2vid_output_imgs):
                if i < len(h5_timestamps):
                    frame_info = {
                        'filename': img_file.name,
                        'filepath': str(img_file.relative_to(output_path)),
                        'frame_index': i,
                        'timestamp_s': float(h5_timestamps[i] / 1e9),
                        'timestamp_ns': int(h5_timestamps[i]),
                        'source_route': 'h5_direct',
                        'model': model_name,
                        'route_fidelity': 'approximate_voxel_to_pseudo_event',
                        'window_duration_ms': window_duration_ms
                    }
                    manifest_frames.append(frame_info)

            # Save frame manifest
            frame_manifest = {
                'reconstruction_method': model_name,
                'source_route': 'h5_direct',
                'route_fidelity': 'approximate_voxel_to_pseudo_event',
                'total_frames': len(manifest_frames),
                'h5_file': str(h5_file),
                'window_duration_ms': window_duration_ms,
                'frames': manifest_frames
            }

            manifest_file = output_path / f"{model_name}_h5_frame_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(frame_manifest, f, indent=2)

            print(f"Frame manifest saved to: {manifest_file}")

        # Save summary
        summary = {
            'h5_file': str(h5_file),
            'model': model_name,
            'output_dir': str(output_path),
            'max_frames': max_frames,
            'window_duration_ms': window_duration_ms,
            'event_count': event_count,
            'success': success,
            'route_fidelity': 'approximate_voxel_to_pseudo_event',
            'note': 'H5 voxel->event conversion is approximate - not equivalent to raw bag events',
            'pseudo_event_thresholds': {'positive': 0.1, 'negative': -0.1},
            'temporal_bin_spread_ms': 0.25 * 5  # 5 bins over 250ms window
        }

        summary_file = output_path / f"{model_name}_h5_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return success

    finally:
        # Clean up temp file
        if os.path.exists(temp_event_file):
            os.unlink(temp_event_file)

def main():
    parser = argparse.ArgumentParser(description="H5 direct reconstruction")
    parser.add_argument("h5_file", help="Path to H5 file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--model", choices=['e2vid', 'firenet'], default='e2vid',
                       help="Model to use")
    parser.add_argument("--max-frames", type=int, help="Max frames to process")
    parser.add_argument("--window-duration", type=float, default=250,
                       help="Window duration in milliseconds")

    args = parser.parse_args()

    if not os.path.exists(args.h5_file):
        print(f"Error: H5 file not found: {args.h5_file}")
        sys.exit(1)

    success = run_h5_reconstruction(
        h5_file=args.h5_file,
        output_dir=args.output,
        model_name=args.model,
        max_frames=args.max_frames,
        window_duration_ms=args.window_duration
    )

    if not success:
        print("H5 reconstruction failed!")
        sys.exit(1)
    else:
        print("H5 reconstruction completed successfully!")

if __name__ == "__main__":
    main()