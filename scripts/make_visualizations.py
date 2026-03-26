#!/usr/bin/env python3
"""
Create visualizations and comparison figures for reconstruction methods.

This script generates side-by-side comparison panels, metric plots,
and sample outputs for the event-to-RGB reconstruction pipeline.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import json
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns

def load_h5_references(h5_file, max_frames=10):
    """Load RGB reference images from H5 file."""
    print(f"Loading RGB references from {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        # Check if rgb_indices exists (preferred)
        if 'rgb_indices' in f:
            rgb_images = f['rgb_images'][...]  # All RGB images
            timestamps_ns = f['timestamps_ns'][...]  # All timestamps
            rgb_indices = f['rgb_indices'][...]  # Indices mapping

            # Limit frames for visualization
            if max_frames and len(rgb_images) > max_frames:
                rgb_images = rgb_images[:max_frames]
                rgb_indices = rgb_indices[:max_frames]

            # Get timestamps using rgb_indices
            ref_timestamps = timestamps_ns[rgb_indices]
            print(f"Loaded {len(rgb_images)} RGB reference frames using rgb_indices")
            return rgb_images, ref_timestamps
        else:
            print("Warning: rgb_indices not found, falling back to rgb_mask")
            rgb_images = f['rgb_images'][...][:max_frames]  # Limit for visualization
            timestamps_ns = f['timestamps_ns'][...][:max_frames]
            rgb_mask = f['rgb_mask'][...][:max_frames]

            # Only use frames with valid RGB data
            valid_indices = np.where(rgb_mask)[0]
            if len(valid_indices) == 0:
                print("No valid RGB frames found!")
                return None, None

            rgb_images = rgb_images[valid_indices]
            timestamps_ns = timestamps_ns[valid_indices]
            print(f"Loaded {len(rgb_images)} RGB reference frames using rgb_mask (fallback)")
            return rgb_images, timestamps_ns

def load_reconstruction_images(recon_dir, method_name, max_frames=10):
    """Load reconstruction images from a directory."""
    print(f"Loading {method_name} reconstructions from {recon_dir}")

    recon_path = Path(recon_dir)

    # Find PNG files
    png_files = list(recon_path.glob("*.png"))
    if not png_files:
        # Try subdirectories (E2VID creates subdirectories)
        subdirs = [d for d in recon_path.iterdir() if d.is_dir()]
        for subdir in subdirs:
            png_files.extend(list(subdir.glob("*.png")))

    png_files.sort()
    png_files = png_files[:max_frames]  # Limit for visualization

    if not png_files:
        print(f"No PNG files found in {recon_dir}")
        return None, None

    # Load images
    images = []
    timestamps = []

    for img_file in png_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            # Extract timestamp from filename if possible
            try:
                # Try to parse timestamp from filename
                if '_t' in img_file.stem:
                    ts_str = img_file.stem.split('_t')[1].split('_')[0]
                    timestamp = float(ts_str)
                else:
                    timestamp = len(timestamps) * 0.25  # Default 250ms intervals
                timestamps.append(timestamp)
            except:
                timestamp = len(timestamps) * 0.25
                timestamps.append(timestamp)

    print(f"Loaded {len(images)} {method_name} reconstructions")
    return images, timestamps

def create_comparison_panel(reference_rgb, recon_e2vid, recon_h5, recon_timesurface,
                          frame_idx=0, timestamp_s=0.0):
    """Create a comparison panel showing all methods side by side."""

    # Convert reference to grayscale for comparison
    if len(reference_rgb.shape) == 3:
        ref_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    else:
        ref_gray = reference_rgb

    # Resize all to common size (720x1280)
    target_size = (1280, 720)

    ref_resized = cv2.resize(ref_gray, target_size)

    # Handle None cases
    if recon_e2vid is not None:
        e2vid_resized = cv2.resize(recon_e2vid, target_size)
    else:
        e2vid_resized = np.zeros((720, 1280), dtype=np.uint8)

    if recon_h5 is not None:
        h5_resized = cv2.resize(recon_h5, target_size)
    else:
        h5_resized = np.zeros((720, 1280), dtype=np.uint8)

    if recon_timesurface is not None:
        ts_resized = cv2.resize(recon_timesurface, target_size)
    else:
        ts_resized = np.zeros((720, 1280), dtype=np.uint8)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Event-to-RGB Reconstruction Comparison\nFrame {frame_idx}, t={timestamp_s:.3f}s', fontsize=16)

    # Top row: Methods
    axes[0, 0].imshow(ref_resized, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Reference RGB (grayscale)', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(e2vid_resized, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('E2VID Bag-Direct', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(h5_resized, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('E2VID H5-Direct', fontsize=12)
    axes[0, 2].axis('off')

    # Bottom row: More methods and error
    axes[1, 0].imshow(ts_resized, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Time-Surface Baseline', fontsize=12)
    axes[1, 0].axis('off')

    # Error maps
    if recon_e2vid is not None:
        error_e2vid = np.abs(ref_resized.astype(float) - e2vid_resized.astype(float))
        im1 = axes[1, 1].imshow(error_e2vid, cmap='hot', vmin=0, vmax=128)
        axes[1, 1].set_title('E2VID Error (|ref - recon|)', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im1, ax=axes[1, 1], shrink=0.7)

    if recon_timesurface is not None:
        error_ts = np.abs(ref_resized.astype(float) - ts_resized.astype(float))
        im2 = axes[1, 2].imshow(error_ts, cmap='hot', vmin=0, vmax=128)
        axes[1, 2].set_title('Time-Surface Error', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im2, ax=axes[1, 2], shrink=0.7)

    plt.tight_layout()
    return fig

def create_metrics_plot(evaluation_results):
    """Create bar plots comparing metrics across methods."""

    if not evaluation_results:
        print("No evaluation results to plot")
        return None

    # Extract metrics
    methods = []
    psnr_vals = []
    ssim_vals = []

    for method, data in evaluation_results.items():
        if 'aggregate_metrics' in data:
            metrics = data['aggregate_metrics']
            methods.append(method.replace('_', ' ').title())
            psnr_vals.append(metrics.get('psnr_mean', 0))
            ssim_vals.append(metrics.get('ssim_mean', 0))

    if not methods:
        return None

    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR comparison
    bars1 = ax1.bar(methods, psnr_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Peak Signal-to-Noise Ratio')
    ax1.set_ylim(0, max(psnr_vals) * 1.1 if psnr_vals else 30)

    # Add value labels on bars
    for bar, val in zip(bars1, psnr_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(psnr_vals)*0.01,
                f'{val:.1f}', ha='center', va='bottom')

    # SSIM comparison
    bars2 = ax2.bar(methods, ssim_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
    ax2.set_ylabel('SSIM')
    ax2.set_title('Structural Similarity Index')
    ax2.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, val in zip(bars2, ssim_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')

    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

def generate_sample_outputs(
    h5_file,
    reconstruction_dirs,
    output_dir,
    max_samples=3
):
    """
    Generate sample comparison outputs.

    Args:
        h5_file: Path to H5 reference file
        reconstruction_dirs: Dict of {method_name: directory_path}
        output_dir: Output directory for figures
        max_samples: Number of sample comparisons to generate
    """

    print(f"=== Generating Sample Outputs ===")

    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    samples_dir = output_path / "samples"
    figures_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Load reference images
    rgb_refs, ref_timestamps = load_h5_references(h5_file, max_frames=max_samples)
    if rgb_refs is None:
        print("Could not load reference images")
        return False

    # Load reconstruction images for each method
    reconstructions = {}
    for method, recon_dir in reconstruction_dirs.items():
        if os.path.exists(recon_dir):
            images, timestamps = load_reconstruction_images(recon_dir, method, max_frames=max_samples)
            reconstructions[method] = images
        else:
            print(f"Reconstruction directory not found: {recon_dir}")
            reconstructions[method] = None

    # Generate comparison panels
    for i in range(min(max_samples, len(rgb_refs))):
        print(f"Creating comparison panel {i+1}/{max_samples}")

        # Get reference
        ref_rgb = rgb_refs[i]

        # Get reconstructions
        e2vid_recon = reconstructions.get('e2vid_bag_direct', [None])[i] if reconstructions.get('e2vid_bag_direct') else None
        h5_recon = reconstructions.get('e2vid_h5_direct', [None])[i] if reconstructions.get('e2vid_h5_direct') else None
        ts_recon = reconstructions.get('timesurface', [None])[i] if reconstructions.get('timesurface') else None

        # Create comparison panel
        fig = create_comparison_panel(
            reference_rgb=ref_rgb,
            recon_e2vid=e2vid_recon,
            recon_h5=h5_recon,
            recon_timesurface=ts_recon,
            frame_idx=i,
            timestamp_s=i * 0.25  # Approximate timestamp
        )

        # Save figure
        panel_file = figures_dir / f"comparison_panel_{i:03d}.png"
        fig.savefig(panel_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved comparison panel: {panel_file}")

    return True

def create_method_overview(output_dir):
    """Create a method overview figure showing the pipeline."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Create a flow chart showing the different routes
    # This is a simple text-based visualization

    ax.text(0.5, 0.9, 'Event-to-RGB Reconstruction Pipeline',
           horizontalalignment='center', fontsize=16, fontweight='bold', transform=ax.transAxes)

    # Route boxes
    routes = [
        ('Raw Events\n(ROS Bag)', 0.1, 0.7, '#e1f5fe'),
        ('H5 Voxels\n(Pre-computed)', 0.1, 0.5, '#f3e5f5'),
        ('Time Surface\n(Baseline)', 0.1, 0.3, '#e8f5e8')
    ]

    methods = [
        ('E2VID\nReconstruction', 0.5, 0.6, '#fff3e0'),
        ('FireNet\nReconstruction', 0.5, 0.4, '#fff3e0')
    ]

    outputs = [
        ('Intensity\nImages', 0.85, 0.7, '#fce4ec'),
        ('Quality\nMetrics', 0.85, 0.5, '#fce4ec'),
        ('Comparisons', 0.85, 0.3, '#fce4ec')
    ]

    # Draw boxes and arrows
    all_boxes = routes + methods + outputs

    for text, x, y, color in all_boxes:
        # Create rectangle
        rect = mpatches.FancyBboxPatch((x-0.08, y-0.05), 0.16, 0.1,
                                      boxstyle="round,pad=0.01",
                                      facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        # Add text
        ax.text(x, y, text, horizontalalignment='center', verticalalignment='center',
               fontsize=10, transform=ax.transAxes)

    # Add arrows (simplified)
    arrow_props = dict(arrowstyle='->', lw=1.5, color='gray')

    # Routes to methods
    ax.annotate('', xy=(0.42, 0.6), xytext=(0.18, 0.7),
                arrowprops=arrow_props, transform=ax.transAxes)
    ax.annotate('', xy=(0.42, 0.6), xytext=(0.18, 0.5),
                arrowprops=arrow_props, transform=ax.transAxes)

    # Time surface to output
    ax.annotate('', xy=(0.77, 0.5), xytext=(0.18, 0.3),
                arrowprops=arrow_props, transform=ax.transAxes)

    # Methods to outputs
    ax.annotate('', xy=(0.77, 0.6), xytext=(0.58, 0.6),
                arrowprops=arrow_props, transform=ax.transAxes)
    ax.annotate('', xy=(0.77, 0.4), xytext=(0.58, 0.4),
                arrowprops=arrow_props, transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    overview_file = figures_dir / "method_overview.png"
    fig.savefig(overview_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved method overview: {overview_file}")
    return overview_file

def main():
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--h5-reference", required=True,
                       help="Path to H5 file with RGB references")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for figures")
    parser.add_argument("--e2vid-bag-dir",
                       help="E2VID bag-direct reconstruction directory")
    parser.add_argument("--e2vid-h5-dir",
                       help="E2VID H5-direct reconstruction directory")
    parser.add_argument("--timesurface-dir",
                       help="Time-surface reconstruction directory")
    parser.add_argument("--evaluation-results",
                       help="JSON file with evaluation results")
    parser.add_argument("--max-samples", type=int, default=3,
                       help="Maximum number of sample comparisons")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.h5_reference):
        print(f"Error: H5 reference file not found: {args.h5_reference}")
        sys.exit(1)

    print("=== Event-to-RGB Visualization Generator ===")

    # Create method overview
    create_method_overview(args.output)

    # Prepare reconstruction directories
    reconstruction_dirs = {}

    if args.e2vid_bag_dir and os.path.exists(args.e2vid_bag_dir):
        reconstruction_dirs['e2vid_bag_direct'] = args.e2vid_bag_dir

    if args.e2vid_h5_dir and os.path.exists(args.e2vid_h5_dir):
        reconstruction_dirs['e2vid_h5_direct'] = args.e2vid_h5_dir

    if args.timesurface_dir and os.path.exists(args.timesurface_dir):
        reconstruction_dirs['timesurface'] = args.timesurface_dir

    if reconstruction_dirs:
        # Generate sample comparison panels
        success = generate_sample_outputs(
            h5_file=args.h5_reference,
            reconstruction_dirs=reconstruction_dirs,
            output_dir=args.output,
            max_samples=args.max_samples
        )

        if success:
            print("✓ Sample visualization panels generated successfully")
        else:
            print("✗ Sample visualization generation failed")
    else:
        print("No valid reconstruction directories provided for comparisons")

    # Generate metrics plot if evaluation results provided
    if args.evaluation_results and os.path.exists(args.evaluation_results):
        try:
            with open(args.evaluation_results, 'r') as f:
                eval_data = json.load(f)

            metrics_fig = create_metrics_plot(eval_data)
            if metrics_fig:
                output_path = Path(args.output)
                figures_dir = output_path / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)

                metrics_file = figures_dir / "metrics_comparison.png"
                metrics_fig.savefig(metrics_file, dpi=150, bbox_inches='tight')
                plt.close(metrics_fig)

                print(f"✓ Metrics comparison plot saved: {metrics_file}")
        except Exception as e:
            print(f"Could not generate metrics plot: {e}")

    print(f"✓ Visualization generation complete! Check: {args.output}/figures/")

if __name__ == "__main__":
    main()