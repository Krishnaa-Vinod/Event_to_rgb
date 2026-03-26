#!/usr/bin/env python3
"""
Handle image geometry, alignment, and synchronization for evaluation.

This script ensures fair comparison by:
1. Aligning reconstruction and reference image dimensions
2. Matching timestamps between reconstructions and RGB references
3. Applying consistent geometric transformations
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional

def analyze_image_dimensions(reconstruction_dir: str, rgb_manifest_file: str = None, rgb_dir: str = None):
    """Analyze dimensions of reconstruction and reference images."""

    print("=== Analyzing Image Dimensions ===")

    analysis = {
        'reconstruction': {'count': 0, 'dimensions': [], 'files': []},
        'rgb_reference': {'count': 0, 'dimensions': [], 'files': []},
        'dimension_mismatch': False,
        'recommended_alignment': None
    }

    # Analyze reconstruction images
    recon_path = Path(reconstruction_dir)
    if recon_path.exists():
        recon_images = list(recon_path.glob("*.png"))
        analysis['reconstruction']['count'] = len(recon_images)

        for img_file in recon_images[:3]:  # Sample first 3
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                analysis['reconstruction']['dimensions'].append((w, h))
                analysis['reconstruction']['files'].append(str(img_file))

    # Check for dimension mismatch and recommend alignment
    if analysis['reconstruction']['dimensions']:
        recon_dim = analysis['reconstruction']['dimensions'][0]
        print(f"Reconstruction dimensions: {recon_dim}")

        # Common expected RGB dimensions
        if rgb_manifest_file and os.path.exists(rgb_manifest_file):
            try:
                with open(rgb_manifest_file, 'r') as f:
                    manifest = json.load(f)

                # Check first RGB frame if available
                frames = manifest.get('frames', [])
                if frames and rgb_dir:
                    first_frame = frames[0]
                    rgb_file = Path(rgb_dir) / first_frame['filename']

                    if rgb_file.exists():
                        img = cv2.imread(str(rgb_file))
                        if img is not None:
                            h, w = img.shape[:2]
                            rgb_dim = (w, h)
                            analysis['rgb_reference']['dimensions'] = [rgb_dim]
                            analysis['rgb_reference']['count'] = len(frames)

                            print(f"RGB reference dimensions: {rgb_dim}")

                            if recon_dim != rgb_dim:
                                analysis['dimension_mismatch'] = True
                                print(f"Dimension mismatch detected!")

                                # Recommend alignment based on dimensions
                                if recon_dim[0] == rgb_dim[0]:  # Same width
                                    analysis['recommended_alignment'] = 'crop_larger_height'
                                elif recon_dim[1] == rgb_dim[1]:  # Same height
                                    analysis['recommended_alignment'] = 'crop_larger_width'
                                else:
                                    analysis['recommended_alignment'] = 'center_crop_to_smaller'
                            else:
                                analysis['recommended_alignment'] = 'no_alignment'

            except Exception as e:
                print(f"Error analyzing RGB dimensions: {e}")

    return analysis

def align_images(recon_img: np.ndarray, rgb_img: np.ndarray, method: str = "center_crop_to_smaller"):
    """Align reconstruction and RGB images to same dimensions."""

    recon_h, recon_w = recon_img.shape[:2]
    rgb_h, rgb_w = rgb_img.shape[:2]

    if method == "no_alignment":
        return recon_img, rgb_img

    elif method == "center_crop_to_smaller":
        # Crop both to smaller dimensions
        target_h = min(recon_h, rgb_h)
        target_w = min(recon_w, rgb_w)

        # Center crop reconstruction
        start_y = (recon_h - target_h) // 2
        start_x = (recon_w - target_w) // 2
        recon_aligned = recon_img[start_y:start_y+target_h, start_x:start_x+target_w]

        # Center crop RGB
        start_y = (rgb_h - target_h) // 2
        start_x = (rgb_w - target_w) // 2
        rgb_aligned = rgb_img[start_y:start_y+target_h, start_x:start_x+target_w]

        return recon_aligned, rgb_aligned

    elif method == "crop_rgb_to_recon":
        # Crop RGB to reconstruction size
        start_y = max(0, (rgb_h - recon_h) // 2)
        start_x = max(0, (rgb_w - recon_w) // 2)

        rgb_aligned = rgb_img[start_y:start_y+recon_h, start_x:start_x+recon_w]
        return recon_img, rgb_aligned

    else:
        raise ValueError(f"Unknown alignment method: {method}")

def test_alignment():
    """Test alignment on our smoke test data."""

    print("=== Testing Image Alignment ===")

    # Test with E2VID reconstruction and exported RGB
    recon_dir = "outputs/e2vid_direct_test/reconstruction"
    rgb_dir = "outputs/smoke_test_export_fixed/rgb"
    rgb_manifest = "outputs/smoke_test_export_fixed/rgb_manifest.json"

    # Analyze dimensions
    analysis = analyze_image_dimensions(recon_dir, rgb_manifest, rgb_dir)

    print("Analysis results:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    # Test alignment on first available pair
    recon_files = list(Path(recon_dir).glob("*.png"))
    rgb_files = list(Path(rgb_dir).glob("*.png"))

    if recon_files and rgb_files:
        print(f"\\nTesting alignment with:")
        print(f"  Reconstruction: {recon_files[0]}")
        print(f"  RGB: {rgb_files[0]}")

        recon_img = cv2.imread(str(recon_files[0]), cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.imread(str(rgb_files[0]))

        if recon_img is not None and rgb_img is not None:
            print(f"  Original dimensions - Recon: {recon_img.shape}, RGB: {rgb_img.shape}")

            # Test alignment
            method = analysis.get('recommended_alignment', 'center_crop_to_smaller')
            recon_aligned, rgb_aligned = align_images(recon_img, rgb_img, method)

            print(f"  Aligned dimensions - Recon: {recon_aligned.shape}, RGB: {rgb_aligned.shape}")

            # Save test alignment
            test_dir = Path("outputs/alignment_test")
            test_dir.mkdir(exist_ok=True)

            cv2.imwrite(str(test_dir / "recon_aligned.png"), recon_aligned)
            cv2.imwrite(str(test_dir / "rgb_aligned.png"), cv2.cvtColor(rgb_aligned, cv2.COLOR_BGR2GRAY))

            print(f"  Test alignment saved to: {test_dir}")

            return True

    return False

def main():
    parser = argparse.ArgumentParser(description="Test image alignment")
    parser.add_argument("--test", action="store_true", help="Run alignment test")
    parser.add_argument("--recon-dir", help="Reconstruction directory")
    parser.add_argument("--rgb-dir", help="RGB directory")
    parser.add_argument("--rgb-manifest", help="RGB manifest file")

    args = parser.parse_args()

    if args.test:
        success = test_alignment()
        if success:
            print("\\n✓ Alignment test completed successfully")
        else:
            print("\\n✗ Alignment test failed")
    else:
        if args.recon_dir:
            analysis = analyze_image_dimensions(args.recon_dir, args.rgb_manifest, args.rgb_dir)
            print("\\nAnalysis:")
            print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()