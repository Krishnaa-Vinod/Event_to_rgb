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
python scripts/run_all.py --config configs/defaults.yaml

# Run on a smoke-test subset
python scripts/run_all.py --config configs/defaults.yaml --smoke-test
```

### Individual components
```bash
# Inspect input data
python scripts/inspect_inputs.py

# Export events from bags
python scripts/export_events_from_bag.py

# Run reconstructions
python scripts/run_bag_reconstruction.py
python scripts/run_h5_reconstruction.py
python scripts/time_surface_baseline.py

# Evaluate results
python scripts/evaluate_reconstructions.py

# Generate visualizations
python scripts/make_visualizations.py
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
- `results/reports/`: CSV/JSON metrics and leaderboards
- `results/figures/`: Plots and comparisons
- `results/samples/`: Representative output images

## Technical Details

See the documentation for implementation details:
- [docs/METHOD.md](docs/METHOD.md): Technical design and data routes
- [docs/ASSUMPTIONS_AND_LIMITATIONS.md](docs/ASSUMPTIONS_AND_LIMITATIONS.md): Important caveats
- [docs/RESULTS.md](docs/RESULTS.md): Performance comparison

## Citation

Based on:
- E2VID: [Rebecq et al., "High Speed and High Dynamic Range Video with an Event Camera", T-PAMI 2021]
- FireNet: [Scheerlinck et al., "Fast Image Reconstruction with an Event Camera", WACV 2020]