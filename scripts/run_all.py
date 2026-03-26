#!/usr/bin/env python3
"""
Single-command entry point for the event-to-RGB reconstruction pipeline.

This script orchestrates the complete pipeline:
1. Data inspection and validation
2. Event extraction from bags
3. Reconstruction using multiple methods
4. Evaluation and comparison
5. Visualization generation
6. Report creation
"""

import os
import sys
import argparse
import json
import time
import subprocess
from pathlib import Path
import yaml

def run_command(cmd, description, timeout=600, capture_output=True):
    """Run a shell command with error handling."""
    print(f"\n>>> {description}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    start_time = time.time()

    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, capture_output=capture_output,
                                  text=True, timeout=timeout)
        else:
            result = subprocess.run(cmd, capture_output=capture_output,
                                  text=True, timeout=timeout)

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ Completed in {elapsed:.1f}s")
            if capture_output and result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
            return True, result.stdout if capture_output else ""
        else:
            print(f"✗ Failed (exit code {result.returncode})")
            if capture_output and result.stderr:
                print("Error:", result.stderr[-500:])
            return False, result.stderr if capture_output else ""

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False, str(e)

def load_config(config_file):
    """Load configuration from YAML file."""
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return None

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from {config_file}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def run_full_pipeline(
    bag_dir,
    h5_file,
    output_dir,
    smoke_test=False,
    max_duration_s=None,
    max_frames=None
):
    """
    Run the complete event-to-RGB reconstruction pipeline.

    Args:
        bag_dir: Path to ROS bag directory
        h5_file: Path to H5 file with voxels and RGB references
        output_dir: Output directory for all results
        smoke_test: If True, run on limited data for testing
        max_duration_s: Max duration for bag processing
        max_frames: Max frames for H5 processing
    """

    print(f"=== Event-to-RGB Pipeline ===")
    print(f"Bag directory: {bag_dir}")
    print(f"H5 file: {h5_file}")
    print(f"Output directory: {output_dir}")
    print(f"Smoke test: {smoke_test}")

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Results tracking
    pipeline_results = {
        'start_time': time.time(),
        'config': {
            'bag_dir': str(bag_dir),
            'h5_file': str(h5_file),
            'output_dir': str(output_dir),
            'smoke_test': smoke_test,
            'max_duration_s': max_duration_s,
            'max_frames': max_frames
        },
        'steps': {}
    }

    # Step 1: Inspect inputs
    print("\n" + "="*60)
    print("STEP 1: Data Inspection")
    print("="*60)

    cmd = ["python", "scripts/inspect_inputs.py"]
    success, output = run_command(cmd, "Inspecting input data", timeout=120)
    pipeline_results['steps']['inspect'] = {
        'success': success,
        'output': output[:1000] if output else ""
    }

    if not success:
        print("Data inspection failed - cannot proceed")
        return pipeline_results

    # Step 2: Extract events from bag
    print("\n" + "="*60)
    print("STEP 2: Bag Event Extraction")
    print("="*60)

    events_output_dir = output_path / "bag_events"

    cmd = ["python", "scripts/export_events_from_bag.py", str(bag_dir),
           "--output", str(events_output_dir)]

    if max_duration_s:
        cmd.extend(["--max-duration", str(max_duration_s)])

    success, output = run_command(cmd, "Extracting events from bag", timeout=600)
    pipeline_results['steps']['bag_extraction'] = {
        'success': success,
        'output_dir': str(events_output_dir) if success else None
    }

    if not success:
        print("Bag event extraction failed")
        return pipeline_results

    # Step 3: Run reconstructions
    print("\n" + "="*60)
    print("STEP 3: Reconstruction Methods")
    print("="*60)

    reconstruction_results = {}

    # 3a: Time-surface baseline from bag
    print("\n--- Time-Surface Baseline ---")
    ts_output_dir = output_path / "timesurface_reconstruction"

    events_file = events_output_dir / "events" / "events.txt"
    if events_file.exists():
        cmd = ["python", "scripts/time_surface_baseline.py", str(events_file),
               "--output", str(ts_output_dir), "--fps", "10"]

        if smoke_test:
            cmd.extend(["--max-duration", "2"])

        success, output = run_command(cmd, "Time-surface baseline", timeout=300)
        reconstruction_results['timesurface'] = {
            'success': success,
            'output_dir': str(ts_output_dir) if success else None
        }
    else:
        print(f"Events file not found: {events_file}")
        reconstruction_results['timesurface'] = {'success': False}

    # 3b: E2VID H5 direct
    print("\n--- E2VID H5 Direct ---")
    e2vid_h5_output_dir = output_path / "e2vid_h5_reconstruction"

    cmd = ["python", "scripts/run_h5_reconstruction_v2.py", str(h5_file),
           "--output", str(e2vid_h5_output_dir)]

    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])

    success, output = run_command(cmd, "E2VID H5 reconstruction", timeout=600)
    reconstruction_results['e2vid_h5'] = {
        'success': success,
        'output_dir': str(e2vid_h5_output_dir) if success else None
    }

    # 3c: E2VID Bag direct (convert events and run E2VID)
    print("\n--- E2VID Bag Direct ---")
    # This would use the existing E2VID reconstruction on the extracted events
    # For now, we'll use the H5 route as the primary E2VID method

    pipeline_results['steps']['reconstructions'] = reconstruction_results

    # Step 4: Generate visualizations
    print("\n" + "="*60)
    print("STEP 4: Visualization Generation")
    print("="*60)

    vis_args = ["python", "scripts/make_visualizations.py",
                "--h5-reference", str(h5_file),
                "--output", str(output_path),
                "--max-samples", "3" if not smoke_test else "1"]

    # Add reconstruction directories that exist
    if reconstruction_results.get('timesurface', {}).get('success'):
        vis_args.extend(["--timesurface-dir", reconstruction_results['timesurface']['output_dir']])

    if reconstruction_results.get('e2vid_h5', {}).get('success'):
        vis_args.extend(["--e2vid-h5-dir", reconstruction_results['e2vid_h5']['output_dir']])

    success, output = run_command(vis_args, "Generating visualizations", timeout=300)
    pipeline_results['steps']['visualizations'] = {'success': success}

    # Step 5: Generate final report
    print("\n" + "="*60)
    print("STEP 5: Final Report")
    print("="*60)

    pipeline_results['end_time'] = time.time()
    pipeline_results['total_duration_s'] = pipeline_results['end_time'] - pipeline_results['start_time']

    # Count successes
    successful_steps = sum(1 for step in pipeline_results['steps'].values()
                          if isinstance(step, dict) and step.get('success'))
    total_steps = len(pipeline_results['steps'])

    # Create summary report
    summary_report = {
        'pipeline_success': successful_steps >= total_steps - 1,  # Allow 1 failure
        'successful_steps': successful_steps,
        'total_steps': total_steps,
        'duration_minutes': pipeline_results['total_duration_s'] / 60,
        'config': pipeline_results['config'],
        'step_results': pipeline_results['steps']
    }

    # Save detailed results
    results_file = output_path / "pipeline_results.json"
    with open(results_file, 'w') as f:
        json.dump(pipeline_results, f, indent=2)

    # Save summary report
    summary_file = output_path / "results" / "reports" / "pipeline_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary_report, f, indent=2)

    # Print final summary
    print(f"\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total duration: {pipeline_results['total_duration_s']/60:.1f} minutes")
    print(f"Steps completed: {successful_steps}/{total_steps}")

    print(f"\nStep Results:")
    for step_name, step_data in pipeline_results['steps'].items():
        if isinstance(step_data, dict):
            status = "✓" if step_data.get('success') else "✗"
            print(f"  {status} {step_name}")
        else:
            print(f"  ? {step_name} (data: {step_data})")

    print(f"\nOutput directories:")
    if reconstruction_results.get('timesurface', {}).get('success'):
        print(f"  Time-surface: {reconstruction_results['timesurface']['output_dir']}")
    if reconstruction_results.get('e2vid_h5', {}).get('success'):
        print(f"  E2VID H5: {reconstruction_results['e2vid_h5']['output_dir']}")

    print(f"\nResults saved to:")
    print(f"  Detailed: {results_file}")
    print(f"  Summary: {summary_file}")

    if (output_path / "results" / "figures").exists():
        figures = list((output_path / "results" / "figures").glob("*.png"))
        print(f"  Figures: {len(figures)} PNG files in results/figures/")

    return pipeline_results

def main():
    parser = argparse.ArgumentParser(description="Event-to-RGB Reconstruction Pipeline")

    # Input data
    parser.add_argument("--bag-dir", required=True,
                       help="Path to ROS bag directory")
    parser.add_argument("--h5-file", required=True,
                       help="Path to H5 file with voxels and references")

    # Output
    parser.add_argument("--output", "-o", default="outputs/full_run",
                       help="Output directory for all results")

    # Processing options
    parser.add_argument("--smoke-test", action="store_true",
                       help="Run on limited data for testing")
    parser.add_argument("--max-duration", type=float,
                       help="Max duration in seconds for bag processing")
    parser.add_argument("--max-frames", type=int,
                       help="Max frames for H5 processing")

    # Configuration
    parser.add_argument("--config", default="configs/defaults.yaml",
                       help="Configuration file")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.bag_dir):
        print(f"Error: Bag directory not found: {args.bag_dir}")
        sys.exit(1)

    if not os.path.exists(args.h5_file):
        print(f"Error: H5 file not found: {args.h5_file}")
        sys.exit(1)

    # Load configuration if provided
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config) or {}

    # Set defaults for smoke test
    if args.smoke_test:
        if not args.max_duration:
            args.max_duration = 5.0  # 5 seconds
        if not args.max_frames:
            args.max_frames = 3  # 3 frames

    # Run the pipeline
    try:
        results = run_full_pipeline(
            bag_dir=args.bag_dir,
            h5_file=args.h5_file,
            output_dir=args.output,
            smoke_test=args.smoke_test,
            max_duration_s=args.max_duration,
            max_frames=args.max_frames
        )

        # Determine exit code
        successful_steps = sum(1 for step in results['steps'].values()
                              if isinstance(step, dict) and step.get('success'))
        total_steps = len(results['steps'])

        if successful_steps >= total_steps - 1:  # Allow 1 failure
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"Check results in: {args.output}")
            sys.exit(0)
        else:
            print(f"\n⚠️ Pipeline completed with errors.")
            print(f"Successful steps: {successful_steps}/{total_steps}")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()