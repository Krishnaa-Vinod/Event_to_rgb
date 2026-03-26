#!/usr/bin/env python3
"""
Run E2VID and FireNet reconstruction from exported bag events.

This script uses the extracted events.txt file to run reconstruction
using the upstream E2VID and FireNet models.
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
import numpy as np
import cv2

def run_e2vid_reconstruction(
    events_file: str,
    output_dir: str,
    model_path: str,
    window_ms: float = 250.0,
    device: str = "auto"
):
    """
    Run E2VID reconstruction on exported events.

    Args:
        events_file: Path to events.txt file
        output_dir: Output directory for reconstructed frames
        model_path: Path to E2VID model weights
        window_ms: Event window duration in milliseconds
        device: Device to use ('cuda', 'cpu', or 'auto')

    Returns:
        dict: Reconstruction summary
    """

    print(f"=== Running E2VID Reconstruction ===")
    print(f"Events: {events_file}")
    print(f"Model: {model_path}")
    print(f"Window: {window_ms}ms")

    # Setup paths
    e2vid_dir = Path("third_party/rpg_e2vid")
    run_script = e2vid_dir / "run_reconstruction.py"

    if not run_script.exists():
        raise FileNotFoundError(f"E2VID script not found: {run_script}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"E2VID weights not found: {model_path}")

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build command - use absolute paths and correct parameters
    abs_events = os.path.abspath(events_file)
    abs_model = os.path.abspath(model_path)
    abs_output = os.path.abspath(output_path)

    cmd = [
        sys.executable, str(run_script),
        "--input_file", abs_events,
        "--path_to_model", abs_model,
        "--output_folder", abs_output,
        "--fixed_duration",
        "--window_duration", str(window_ms),
        "--no-normalize",
        "--no-recurrent"
    ]

    if device == "cuda":
        cmd.append("--use_gpu")

    print(f"Command: {' '.join(cmd)}")

    # Time the reconstruction
    start_time = time.time()

    try:
        # Run E2VID
        result = subprocess.run(
            cmd,
            cwd=str(e2vid_dir),
            capture_output=True,
            text=True,
            check=True
        )

        duration = time.time() - start_time
        print(f"E2VID completed in {duration:.1f}s")

        # Count output frames and collect metadata
        frame_files = list(output_path.glob("*.png"))
        frame_files.sort()
        frame_count = len(frame_files)

        # Create per-frame manifest
        manifest_frames = []
        for i, frame_file in enumerate(frame_files):
            # Extract timestamp from filename if possible, otherwise use sequence
            try:
                # E2VID typically outputs frame_000001.png etc.
                frame_number = int(frame_file.stem.split('_')[-1])
                # Estimate timestamp based on window duration and frame sequence
                timestamp_s = frame_number * (window_ms / 1000.0)
            except:
                # Fallback to sequence-based timing
                timestamp_s = i * (window_ms / 1000.0)

            frame_info = {
                'filename': frame_file.name,
                'filepath': str(frame_file.relative_to(output_path)),
                'frame_index': i,
                'timestamp_s': float(timestamp_s),
                'source_route': 'bag_direct',
                'model': 'e2vid',
                'route_fidelity': 'exact_raw_event',
                'window_duration_ms': window_ms
            }
            manifest_frames.append(frame_info)

        # Save frame manifest
        frame_manifest = {
            'reconstruction_method': 'e2vid',
            'source_route': 'bag_direct',
            'route_fidelity': 'exact_raw_event',
            'total_frames': len(manifest_frames),
            'events_file': str(events_file),
            'window_duration_ms': window_ms,
            'frames': manifest_frames
        }

        manifest_file = output_path / "e2vid_bag_frame_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(frame_manifest, f, indent=2)

        print(f"Frame manifest saved to: {manifest_file}")

        summary = {
            "method": "e2vid",
            "events_file": str(events_file),
            "model_path": str(model_path),
            "output_dir": str(output_path),
            "window_ms": window_ms,
            "device": device,
            "frame_count": frame_count,
            "duration_s": duration,
            "fps": frame_count / duration if duration > 0 else 0,
            "route_fidelity": "exact_raw_event",
            "manifest_file": str(manifest_file),
            "stdout": result.stdout[-1000:],  # Limit output length
            "stderr": result.stderr[-1000:] if result.stderr else ""
        }

        return summary

    except subprocess.CalledProcessError as e:
        print(f"E2VID failed with return code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

def run_firenet_reconstruction(
    events_file: str,
    output_dir: str,
    model_path: str = None,
    window_ms: float = 250.0,
    device: str = "auto"
):
    """
    Run FireNet reconstruction on exported events.

    Args:
        events_file: Path to events.txt file
        output_dir: Output directory for reconstructed frames
        model_path: Path to FireNet weights (optional)
        window_ms: Event window duration in milliseconds
        device: Device to use ('cuda', 'cpu', or 'auto')

    Returns:
        dict: Reconstruction summary
    """

    print(f"=== Running FireNet Reconstruction ===")
    print(f"Events: {events_file}")
    print(f"Window: {window_ms}ms")

    # Setup paths
    firenet_dir = Path("third_party/rpg_e2vid_firenet")
    run_script = firenet_dir / "run_reconstruction.py"

    if not run_script.exists():
        raise FileNotFoundError(f"FireNet script not found: {run_script}")

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build command - use absolute paths and correct parameters
    abs_events = os.path.abspath(events_file)
    abs_output = os.path.abspath(output_path)

    cmd = [
        sys.executable, str(run_script),
        "--input_file", abs_events,
        "--output_folder", abs_output,
        "--fixed_duration",
        "--window_duration", str(window_ms),
        "--no_normalize",
        "--no_recurrent"
    ]

    if model_path and Path(model_path).exists():
        abs_model = os.path.abspath(model_path)
        cmd.extend(["--path_to_model", abs_model])

    if device == "cuda":
        cmd.append("--use_gpu")

    print(f"Command: {' '.join(cmd)}")

    # Time the reconstruction
    start_time = time.time()

    try:
        # Run FireNet
        result = subprocess.run(
            cmd,
            cwd=str(firenet_dir),
            capture_output=True,
            text=True,
            check=True
        )

        duration = time.time() - start_time
        print(f"FireNet completed in {duration:.1f}s")

        # Count output frames
        frame_files = list(output_path.glob("*.png"))
        frame_count = len(frame_files)

        summary = {
            "method": "firenet",
            "events_file": str(events_file),
            "model_path": str(model_path) if model_path else None,
            "output_dir": str(output_path),
            "window_ms": window_ms,
            "device": device,
            "frame_count": frame_count,
            "duration_s": duration,
            "fps": frame_count / duration if duration > 0 else 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

        return summary

    except subprocess.CalledProcessError as e:
        print(f"FireNet failed with return code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run E2VID/FireNet reconstruction")
    parser.add_argument("methods", nargs="+", choices=["e2vid", "firenet"],
                       help="Reconstruction method(s) to run")
    parser.add_argument("--events", required=True, help="Path to events.txt file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--e2vid-weights", default="weights/E2VID_lightweight.pth.tar",
                       help="Path to E2VID model weights")
    parser.add_argument("--firenet-weights", help="Path to FireNet model weights")
    parser.add_argument("--window-ms", type=float, default=250.0,
                       help="Event window duration in ms")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use for inference")

    args = parser.parse_args()

    if not os.path.exists(args.events):
        print(f"Error: Events file does not exist: {args.events}")
        sys.exit(1)

    results = []

    for method in args.methods:
        try:
            output_dir = Path(args.output) / f"{method}_window_{args.window_ms:.0f}ms"

            if method == "e2vid":
                summary = run_e2vid_reconstruction(
                    events_file=args.events,
                    output_dir=str(output_dir),
                    model_path=args.e2vid_weights,
                    window_ms=args.window_ms,
                    device=args.device
                )
            elif method == "firenet":
                summary = run_firenet_reconstruction(
                    events_file=args.events,
                    output_dir=str(output_dir),
                    model_path=args.firenet_weights,
                    window_ms=args.window_ms,
                    device=args.device
                )

            results.append(summary)
            print(f"\\n{method.upper()} Results:")
            print(f"  Frames: {summary['frame_count']}")
            print(f"  Duration: {summary['duration_s']:.1f}s")
            print(f"  FPS: {summary['fps']:.1f}")

        except Exception as e:
            print(f"\\n{method.upper()} reconstruction failed: {e}")
            results.append({
                "method": method,
                "error": str(e),
                "success": False
            })

    # Save results summary
    results_file = Path(args.output) / "bag_reconstruction_summary.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()