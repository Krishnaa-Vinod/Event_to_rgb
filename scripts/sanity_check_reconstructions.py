#!/usr/bin/env python3
"""
Sanity checks for reconstruction outputs.

This script validates that reconstructions contain meaningful data,
not just empty/black frames.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm

def analyze_image_sanity(image_path):
    """Analyze a single image for sanity metrics."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Compute statistics
    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))

    # Check for near-black pixels (< 10 intensity)
    near_black_fraction = float(np.sum(img < 10) / img.size)

    # Check if the entire image is nearly constant
    is_blank = std_intensity < 2.0 and mean_intensity < 15.0

    return {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'near_black_fraction': near_black_fraction,
        'is_blank': is_blank,
        'resolution': img.shape
    }

def sanity_check_reconstruction(recon_dir, method_name, max_samples=10):
    """Run sanity checks on a reconstruction directory."""

    print(f"\n=== Sanity Check: {method_name.upper()} ===")

    # Find PNG files
    recon_path = Path(recon_dir)
    image_files = list(recon_path.glob("**/*.png"))

    if not image_files:
        return {
            'method': method_name,
            'total_images': 0,
            'error': 'No PNG files found',
            'sanity_verdict': 'INVALID'
        }

    image_files.sort()

    # Sample subset if too many images
    if len(image_files) > max_samples:
        step = len(image_files) // max_samples
        sampled_files = image_files[::step][:max_samples]
    else:
        sampled_files = image_files

    # Analyze each sampled image
    results = []
    valid_analyses = 0

    for img_file in tqdm(sampled_files, desc=f"Analyzing {method_name}"):
        analysis = analyze_image_sanity(img_file)
        if analysis is not None:
            analysis['filename'] = img_file.name
            results.append(analysis)
            valid_analyses += 1

    if not results:
        return {
            'method': method_name,
            'total_images': len(image_files),
            'sampled_images': len(sampled_files),
            'valid_analyses': 0,
            'error': 'Could not analyze any images',
            'sanity_verdict': 'INVALID'
        }

    # Compute aggregate statistics
    mean_intensities = [r['mean_intensity'] for r in results]
    std_intensities = [r['std_intensity'] for r in results]
    near_black_fractions = [r['near_black_fraction'] for r in results]
    blank_count = sum(r['is_blank'] for r in results)

    aggregate_stats = {
        'mean_intensity_avg': float(np.mean(mean_intensities)),
        'mean_intensity_std': float(np.std(mean_intensities)),
        'std_intensity_avg': float(np.mean(std_intensities)),
        'std_intensity_std': float(np.std(std_intensities)),
        'near_black_fraction_avg': float(np.mean(near_black_fractions)),
        'blank_frame_count': blank_count,
        'blank_frame_fraction': float(blank_count / len(results))
    }

    # Determine sanity verdict
    if blank_count / len(results) > 0.5:
        verdict = 'INVALID - Too many blank frames'
    elif aggregate_stats['mean_intensity_avg'] < 5.0:
        verdict = 'SUSPICIOUS - Very low intensity'
    elif aggregate_stats['near_black_fraction_avg'] > 0.8:
        verdict = 'SUSPICIOUS - Mostly dark pixels'
    elif aggregate_stats['std_intensity_avg'] < 3.0:
        verdict = 'SUSPICIOUS - Low variance (flat images)'
    else:
        verdict = 'VALID'

    summary = {
        'method': method_name,
        'reconstruction_dir': str(recon_dir),
        'total_images': len(image_files),
        'sampled_images': len(sampled_files),
        'valid_analyses': valid_analyses,
        'aggregate_stats': aggregate_stats,
        'per_frame_details': results,
        'sanity_verdict': verdict
    }

    print(f"  Total images: {len(image_files)}")
    print(f"  Analyzed: {valid_analyses}")
    print(f"  Mean intensity: {aggregate_stats['mean_intensity_avg']:.1f} ± {aggregate_stats['mean_intensity_std']:.1f}")
    print(f"  Blank frames: {blank_count}/{len(results)} ({aggregate_stats['blank_frame_fraction']:.1%})")
    print(f"  Verdict: {verdict}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Sanity check reconstruction outputs")
    parser.add_argument("--reconstruction-dirs", nargs="+", required=True,
                       help="Directories containing reconstructed images")
    parser.add_argument("--method-names", nargs="+", required=True,
                       help="Method names corresponding to directories")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for sanity check results")
    parser.add_argument("--max-samples", type=int, default=10,
                       help="Max images to sample per method")

    args = parser.parse_args()

    if len(args.reconstruction_dirs) != len(args.method_names):
        print("Error: Number of directories must match number of method names")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run sanity checks
    all_results = []

    for recon_dir, method_name in zip(args.reconstruction_dirs, args.method_names):
        if os.path.exists(recon_dir):
            result = sanity_check_reconstruction(recon_dir, method_name, args.max_samples)
            all_results.append(result)
        else:
            print(f"\nWarning: Directory not found: {recon_dir}")
            all_results.append({
                'method': method_name,
                'error': f'Directory not found: {recon_dir}',
                'sanity_verdict': 'MISSING'
            })

    # Save results
    results_file = output_path / "reconstruction_sanity_checks.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create summary CSV
    import pandas as pd

    summary_rows = []
    for result in all_results:
        if 'aggregate_stats' in result:
            row = {
                'method': result['method'],
                'verdict': result['sanity_verdict'],
                'total_images': result['total_images'],
                'mean_intensity': result['aggregate_stats']['mean_intensity_avg'],
                'std_intensity': result['aggregate_stats']['std_intensity_avg'],
                'blank_fraction': result['aggregate_stats']['blank_frame_fraction'],
                'near_black_fraction': result['aggregate_stats']['near_black_fraction_avg']
            }
        else:
            row = {
                'method': result['method'],
                'verdict': result['sanity_verdict'],
                'total_images': result.get('total_images', 0),
                'mean_intensity': 0,
                'std_intensity': 0,
                'blank_fraction': 1.0,
                'near_black_fraction': 1.0
            }
        summary_rows.append(row)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_file = output_path / "reconstruction_sanity_checks.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Sanity check results saved:")
        print(f"  JSON: {results_file}")
        print(f"  CSV: {csv_file}")

    # Print final summary
    print(f"\n=== SANITY CHECK SUMMARY ===")
    valid_methods = sum(1 for r in all_results if r['sanity_verdict'] == 'VALID')
    total_methods = len(all_results)
    print(f"Valid methods: {valid_methods}/{total_methods}")

    for result in all_results:
        verdict = result['sanity_verdict']
        status_icon = "✓" if verdict == 'VALID' else "⚠️" if 'SUSPICIOUS' in verdict else "✗"
        print(f"  {status_icon} {result['method']}: {verdict}")

if __name__ == "__main__":
    main()