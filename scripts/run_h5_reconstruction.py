#!/usr/bin/env python3
"""
Run E2VID and FireNet reconstruction from H5 voxel tensors.

This script loads pre-computed voxel tensors from H5 files and runs
direct model inference, bypassing the event-to-voxel conversion step.
"""

import os
import sys
import argparse
import h5py
import numpy as np
import torch
import json
import time
from pathlib import Path
import cv2

# Add E2VID paths
sys.path.append('third_party/rpg_e2vid')
sys.path.append('third_party/rpg_e2vid_firenet')

def load_e2vid_model(model_path: str, device: str = "cuda"):
    """Load E2VID model for direct inference."""
    from utils.loading_utils import load_model

    print(f"Loading E2VID model from {model_path}")
    model = load_model(model_path)
    model.eval()
    model.to(device)
    return model

def check_voxel_compatibility(voxel_tensor, expected_shape=(5, 720, 1280)):
    """
    Check if H5 voxels are compatible with E2VID/FireNet preprocessing.

    Returns compatibility info and any required transformations.
    """

    print(f"Voxel tensor shape: {voxel_tensor.shape}")
    print(f"Expected shape: (5, 720, 1280)")
    print(f"Voxel value range: [{voxel_tensor.min():.3f}, {voxel_tensor.max():.3f}]")

    compatibility = {
        "shape_match": voxel_tensor.shape[-3:] == expected_shape,
        "needs_transpose": False,
        "needs_normalization": False,
        "value_range": [float(voxel_tensor.min()), float(voxel_tensor.max())],
        "is_compatible": True
    }

    # Check if normalization is needed (E2VID expects certain ranges)
    if voxel_tensor.min() < -1.1 or voxel_tensor.max() > 1.1:
        compatibility["needs_normalization"] = True

    # Check shape compatibility
    if not compatibility["shape_match"]:
        print(f"Warning: Voxel shape mismatch. May need preprocessing.")
        compatibility["is_compatible"] = False

    return compatibility

def preprocess_voxel_for_inference(voxel_tensor, compatibility_info):
    """
    Preprocess H5 voxel tensor for model inference.
    """

    processed = voxel_tensor.copy()

    # Normalize if needed
    if compatibility_info["needs_normalization"]:
        # Normalize to [-1, 1] range
        v_min, v_max = processed.min(), processed.max()
        if v_max > v_min:
            processed = 2.0 * (processed - v_min) / (v_max - v_min) - 1.0
            print(f"Normalized voxels from [{v_min:.3f}, {v_max:.3f}] to [{processed.min():.3f}, {processed.max():.3f}]")

    return processed

def run_h5_reconstruction(
    h5_file: str,
    output_dir: str,
    method: str = "e2vid",
    model_path: str = None,
    max_frames: int = None,
    device: str = "auto"
):
    """
    Run reconstruction from H5 voxel tensors.
    """

    print(f"=== Running {method.upper()} H5 Reconstruction ===")
    print(f"H5 file: {h5_file}")
    print(f"Output: {output_dir}")

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load H5 data
    print("Loading H5 data...")
    with h5py.File(h5_file, 'r') as f:
        voxels = f['voxels'][:]
        timestamps_ns = f['timestamps_ns'][:]

        print(f"Loaded {len(voxels)} voxel tensors")
        print(f"Voxel shape: {voxels.shape}")
        print(f"Timestamp range: {timestamps_ns[0]} to {timestamps_ns[-1]} ns")

    # Limit frames if requested
    if max_frames:
        voxels = voxels[:max_frames]
        timestamps_ns = timestamps_ns[:max_frames]
        print(f"Limited to {len(voxels)} frames for testing")

    # Check voxel compatibility
    sample_voxel = voxels[0]
    compatibility = check_voxel_compatibility(sample_voxel)

    if not compatibility["is_compatible"]:
        print("Warning: Voxel tensors may not be fully compatible with model expectations")
        print("Proceeding with approximate reconstruction...")

    # Load model
    try:
        if method == "e2vid":
            if not model_path:
                model_path = "weights/E2VID_lightweight.pth.tar"
            model = load_e2vid_model(model_path, device)
        else:
            raise ValueError(f"Method {method} not implemented yet")
    except Exception as e:
        print(f"Error loading {method} model: {e}")
        return {"error": str(e), "success": False}

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run inference on each voxel tensor
    print(f"Running inference on {len(voxels)} voxel tensors...")

    frame_times = []
    reconstructed_frames = []

    start_time = time.time()

    for i, (voxel, timestamp_ns) in enumerate(zip(voxels, timestamps_ns)):

        frame_start = time.time()

        try:
            # Preprocess voxel
            processed_voxel = preprocess_voxel_for_inference(voxel, compatibility)

            # Convert to PyTorch tensor
            voxel_tensor = torch.from_numpy(processed_voxel).float().to(device)

            # Add batch dimension if needed
            if len(voxel_tensor.shape) == 3:
                voxel_tensor = voxel_tensor.unsqueeze(0)  # (1, 5, H, W)

            # Model inference
            with torch.no_grad():
                # Handle recurrent E2VID models that need prev_states
                try:
                    # Try with prev_states=None (for recurrent E2VID)
                    reconstructed = model(voxel_tensor, None)
                except TypeError:
                    # Try without prev_states (for non-recurrent models)
                    reconstructed = model(voxel_tensor)

                # Handle different output formats
                if isinstance(reconstructed, (tuple, list)) and len(reconstructed) >= 2:
                    # E2VID recurrent returns (image, states)
                    frame = reconstructed[0]
                elif isinstance(reconstructed, dict):
                    # Some models might return a dict with 'image' key
                    frame = reconstructed.get('image', reconstructed)
                else:
                    frame = reconstructed

                # Convert to numpy
                if isinstance(frame, torch.Tensor):
                    frame = frame.squeeze().cpu().numpy()
                else:
                    frame = frame

                # Normalize to [0, 255] for saving
                if frame.min() < 0 or frame.max() > 1.5:
                    frame = np.clip((frame - frame.min()) / (frame.max() - frame.min()), 0, 1)
                elif frame.max() <= 1.0:
                    frame = np.clip(frame, 0, 1)
                else:
                    frame = np.clip(frame / 255.0, 0, 1)
                
                frame_uint8 = (frame * 255).astype(np.uint8)

                # Save frame
                timestamp_s = timestamp_ns / 1e9
                frame_filename = f"frame_{i:06d}_t{timestamp_s:.3f}.png"
                frame_path = output_path / frame_filename

                # Ensure proper image format (handle grayscale)
                if len(frame_uint8.shape) == 2:
                    cv2.imwrite(str(frame_path), frame_uint8)
                elif len(frame_uint8.shape) == 3 and frame_uint8.shape[2] == 3:
                    cv2.imwrite(str(frame_path), frame_uint8)
                else:
                    print(f"Unexpected frame shape: {frame_uint8.shape}")
                    continue

                reconstructed_frames.append({
                    'index': i,
                    'filename': frame_filename,
                    'timestamp_ns': int(timestamp_ns),
                    'timestamp_s': float(timestamp_s)
                })

                frame_time = time.time() - frame_start
                frame_times.append(frame_time)

                if (i + 1) % 5 == 0:
                    avg_time = np.mean(frame_times[-5:])
                    print(f"Processed {i + 1}/{len(voxels)} frames, recent avg: {avg_time:.3f}s/frame")

        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    total_time = time.time() - start_time

    print(f"Completed {len(reconstructed_frames)} reconstructions in {total_time:.1f}s")
    if frame_times:
        print(f"Average: {np.mean(frame_times):.3f}s/frame, FPS: {len(reconstructed_frames)/total_time:.1f}")

    # Save reconstruction manifest
    manifest_file = output_path / "h5_reconstruction_manifest.json"
    manifest = {
        'method': method,
        'h5_file': str(h5_file),
        'output_dir': str(output_path),
        'model_path': str(model_path) if model_path else None,
        'frames': reconstructed_frames,
        'compatibility_info': compatibility,
        'timing': {
            'total_time_s': total_time,
            'frames_processed': len(reconstructed_frames),
            'avg_time_per_frame_s': np.mean(frame_times) if frame_times else 0,
            'fps': len(reconstructed_frames) / total_time if total_time > 0 else 0
        }
    }

    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved to: {manifest_file}")

    return manifest

def main():
    parser = argparse.ArgumentParser(description="Run H5 voxel reconstruction")
    parser.add_argument("method", choices=["e2vid", "firenet"], help="Reconstruction method")
    parser.add_argument("--h5-file", required=True, help="Path to H5 file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--model-path", help="Path to model weights")
    parser.add_argument("--max-frames", type=int, help="Max frames to process (for testing)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    if not os.path.exists(args.h5_file):
        print(f"Error: H5 file does not exist: {args.h5_file}")
        sys.exit(1)

    try:
        result = run_h5_reconstruction(
            h5_file=args.h5_file,
            output_dir=args.output,
            method=args.method,
            model_path=args.model_path,
            max_frames=args.max_frames,
            device=args.device
        )

        if result.get("success", True):
            print(f"\n=== {args.method.upper()} H5 Reconstruction Results ===")
            timing = result.get("timing", {})
            print(f"Frames: {timing.get('frames_processed', 0)}")
            print(f"Total time: {timing.get('total_time_s', 0):.1f}s")
            print(f"FPS: {timing.get('fps', 0):.1f}")
        else:
            print(f"H5 reconstruction failed: {result.get('error')}")
            sys.exit(1)

    except Exception as e:
        print(f"H5 reconstruction error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
