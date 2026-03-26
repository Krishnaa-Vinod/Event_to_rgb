# Event-to-RGB Reconstruction Pipeline

This repository implements an end-to-end evaluation pipeline for event-to-intensity reconstruction methods, comparing E2VID, FireNet, and time-surface baselines across different input routes (ROS bags vs. H5 voxel data).

## Overview

This project evaluates event camera reconstruction methods on data from ASU Sol, supporting:
- **ROS bag direct route**: Extract raw events from MCAP/bag files and run reconstruction
- **H5 voxel route**: Use pre-computed voxel tensors for direct model inference
- **Time-surface baseline**: Generate intensity reconstructions from raw events using exponential decay

## Setup

### Environment
This pipeline uses the `nomad-eventvoxels` conda environment (or creates a compatible one):

```bash
# Activate the environment
conda activate nomad-eventvoxels

# Install additional dependencies if needed
pip install -r requirements.txt
```

### Model Weights
Run the setup script to fetch model weights and dependencies:

```bash
./scripts/fetch_models.sh
```

## Usage

### Quick start
```bash
# Run full pipeline on the default dataset
python scripts/run_all.py --bag-dir /scratch/kvinod/bags/overfitting_data/data_collect_20260228_153433 --h5-file /scratch/kvinod/bags/eGo_navi_overfit_data_h5/data_collect_20260228_153433.h5

# Run on a smoke-test subset
python scripts/run_all.py --bag-dir /scratch/kvinod/bags/overfitting_data/data_collect_20260228_153433 --h5-file /scratch/kvinod/bags/eGo_navi_overfit_data_h5/data_collect_20260228_153433.h5 --smoke-test

# Alternative: Use paths configuration file
python scripts/run_all.py --paths-config configs/paths.json
python scripts/run_all.py --paths-config configs/paths.json --smoke-test
```

### Individual components
```bash
# Inspect input data
python scripts/inspect_inputs.py --bag-dir /path/to/bag --h5-file /path/to/file.h5

# Export events from bags
python scripts/export_events_from_bag.py /path/to/bag --output /path/to/output

# Run reconstructions
python scripts/run_bag_reconstruction.py --bag-events /path/to/events --output /path/to/output
python scripts/run_h5_reconstruction_v2.py /path/to/file.h5 --output /path/to/output
python scripts/time_surface_baseline.py /path/to/events.txt --output /path/to/output

# Evaluate results
python scripts/evaluate_reconstructions.py --reconstruction-dir /path/to/recon --reference-h5 /path/to/ref.h5 --method-name method_name --output /path/to/output

# Generate visualizations
python scripts/make_visualizations.py --h5-reference /path/to/ref.h5 --output /path/to/output
```

## Configuration

Copy and edit the example config:
```bash
cp configs/paths.example.json configs/paths.json
```

## Input Data

- **ROS bags**: `/scratch/kvinod/bags/overfitting_data/data_collect_20260228_153433/`
- **H5 voxels**: `/scratch/kvinod/bags/eGo_navi_overfit_data_h5/`

## Results

Results are saved to `results/` with:
- `results/reports/pipeline_summary.json`: Overall pipeline execution status
- `results/reports/leaderboard_fair_250ms.csv`: Primary quantitative leaderboard
- `results/reports/leaderboard_fair_250ms.json`: Leaderboard in JSON format
- `results/reports/per_sequence_metrics.csv`: Per-frame detailed metrics
- `results/figures/`: Comparison plots and visualizations
- `results/samples/`: Representative output images

**Expected result files after successful run**:
- `pipeline_summary.json`: Pipeline execution status and timing
- `leaderboard_fair_250ms.csv`: Main performance comparison table
- `comparison_panel_*.png`: Side-by-side reconstruction comparisons

## Technical Details

This pipeline evaluates event-to-intensity reconstruction methods:
- **Primary quantitative comparison**: Reconstructed grayscale intensity vs grayscale-converted RGB references
- **Bag-direct route**: Uses raw events extracted from MCAP/bag files (exact reconstruction)
- **H5-direct route**: Uses pre-computed voxel tensors converted to pseudo-events (**approximate reconstruction**)
- **Evaluation metrics**: MSE, MAE, PSNR, SSIM on grayscale images with temporal alignment

**Important**: The H5-direct route is marked as approximate because it converts voxel grids back to pseudo-events, which may not exactly match the original event stream.

See the documentation for implementation details:
- [docs/METHOD.md](docs/METHOD.md): Technical design and data routes
- [docs/ASSUMPTIONS_AND_LIMITATIONS.md](docs/ASSUMPTIONS_AND_LIMITATIONS.md): Important caveats
- [docs/RESULTS.md](docs/RESULTS.md): Performance comparison

## Citation

Based on:
- E2VID: [Rebecq et al., "High Speed and High Dynamic Range Video with an Event Camera", T-PAMI 2021]
- FireNet: [Scheerlinck et al., "Fast Image Reconstruction with an Event Camera", WACV 2020]