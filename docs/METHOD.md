# Technical Design and Data Routes

This document describes the technical implementation of the event-to-RGB reconstruction pipeline.

## Overview

The pipeline implements three reconstruction routes:

1. **Bag Direct Route**: Raw events from MCAP/bag files → E2VID/FireNet
2. **H5 Direct Route**: Pre-computed voxel tensors from H5 → E2VID/FireNet
3. **Time-Surface Baseline**: Raw events → exponential decay time surface

## Data Flow Architecture

### Input Data Structure

**ROS Bags (MCAP format):**
- Event topic: `/event_camera/events` (EVT3 encoded, ~40K messages)
- RGB topic: `/cam_sync/cam0/image_raw` (bayer_rggb8, ~3K images)
- Duration: ~108 seconds per bag
- File size: ~8.5 GB MCAP

**H5 Files:**
- `voxels`: (N, 5, 720, 1280) - 5 temporal bins, 250ms windows
- `timestamps_ns`: (N,) - Absolute timestamps per voxel frame
- `rgb_images`: (N, 1024, 1280, 3) - RGB frames (all valid per rgb_mask)
- Event resolution: 720×1280, RGB resolution: 1024×1280

### Route 1: Bag Direct (Raw Events)

**Pipeline:**
1. `scripts/export_events_from_bag.py` - Extract raw events using source pipeline utilities
2. Decode EVT3 format using `eventnavpp_datagen_pipeline/evt3_decoder.py`
3. Export to E2VID format: header line + "time x y polarity" lines
4. `third_party/rpg_e2vid/run_reconstruction.py` - Official E2VID reconstruction

**Key Implementation Details:**
- Uses existing `bag_reader.py` with truncation fallback
- Handles ~25M events per 5-second window
- Converts timestamps from nanoseconds to seconds
- Maintains temporal continuity for model input

**Output Format:**
```
1280 720                    # Header: width height
0.000000000 372 696 0       # time x y polarity
0.000000000 813 256 0
...
```

### Route 2: H5 Direct (Voxel Input)

**Pipeline:**
1. Load pre-computed voxel tensors from H5: (N, 5, 720, 1280)
2. `scripts/run_h5_reconstruction_v2.py` - Convert voxels to pseudo-events
3. Generate sparse events from voxel intensity thresholds
4. Run E2VID reconstruction on pseudo-event stream

**Voxel-to-Event Conversion:**
```python
# For each temporal bin (5 bins per 250ms window)
pos_mask = voxel_bin > 0.1   # Positive events
neg_mask = voxel_bin < -0.1  # Negative events
# Extract coordinates and generate events
```

**Critical Note:** This route is **approximate** since we're reconstructing events from voxels. The H5 voxelization process may not be numerically identical to E2VID's preprocessing.

### Route 3: Time-Surface Baseline

**Algorithm:**
```python
# For each event (t, x, y, pol)
if polarity > 0:
    positive_surface[y, x] = timestamp
else:
    negative_surface[y, x] = timestamp

# Apply exponential decay
decay = exp(-(current_time - timestamp) / tau)
combined = positive_decayed - negative_decayed
```

**Parameters:**
- Tau (decay constant): 50ms default
- Window duration: 250ms (matching H5 voxel windows)
- Output: Grayscale intensity images

## Model Integration

### E2VID Integration
- Uses official `third_party/rpg_e2vid/run_reconstruction.py`
- Weights: `weights/E2VID_lightweight.pth.tar` (41MB)
- Default settings: Fixed duration windows, auto HDR
- Device: CUDA (A100 80GB available)

### FireNet Integration
- Repository: `third_party/rpg_e2vid_firenet` (cedric/firenet branch)
- Weights: Manual download required from Google Drive
- Same interface as E2VID via `run_reconstruction.py`

## Coordinate Systems and Geometry

### Resolution Differences
- **Event data**: 720×1280 (height × width)
- **H5 RGB**: 1024×1280 (height × width)
- **Bag RGB**: bayer_rggb8 format (not yet supported)

### Temporal Alignment
- **Events**: Relative timestamps starting from 0.0s
- **H5 data**: Absolute timestamps (~1.7e12 nanoseconds)
- **Matching window**: ±50ms default tolerance

### Image Preprocessing Pipeline
1. Load reconstruction (grayscale) and reference (RGB)
2. Convert RGB to grayscale for comparison
3. Resize to common resolution (720×1280)
4. Normalize to uint8 [0, 255] range

## Fair Comparison Protocol

### Primary Comparison (Fair)
- **Window policy**: Fixed 250ms duration for all methods
- **Rationale**: Matches H5 voxel window size
- **Methods**: bag_direct_e2vid, bag_direct_firenet, h5_direct_e2vid, timesurface

### Secondary Comparison (Bag-Only)
- **Window policies**: [33ms, 50ms, 100ms, 250ms]
- **Purpose**: Explore optimal bag-direct performance
- **Methods**: bag_direct_e2vid, bag_direct_firenet only

## Quality Metrics

### Image Quality Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

### Temporal Metrics
- **Mean timestamp delta**: Average synchronization error (ms)
- **Median timestamp delta**: Robust sync metric

### Performance Metrics
- **Inference time**: Per-frame processing time (ms)
- **Throughput**: Frames per second (FPS)

## Current Implementation Status

✅ **Completed:**
- Project structure and configuration
- Environment setup (conda + additional packages)
- Data inspection and validation
- Bag direct route (25M events extracted successfully)
- H5 direct route (pseudo-event conversion working)
- Time-surface baseline (exponential decay implementation)
- E2VID model integration
- Basic evaluation framework

⚠️ **Partially Complete:**
- Image geometry alignment (resolution differences handled)
- Temporal synchronization (absolute vs relative timestamps)
- RGB format support (bayer_rggb8 pending)

🔄 **In Progress:**
- Evaluation pipeline refinement
- Comprehensive testing and validation
- Report generation and visualization

## Known Limitations

1. **H5 Route Approximation**: Voxel→event conversion is not exact
2. **Timestamp Alignment**: Absolute vs relative timestamp handling needed
3. **RGB Format**: Bayer pattern decoding not implemented
4. **FireNet**: Manual weight download required
5. **Color Reconstruction**: Current implementation focuses on grayscale intensity

## Dependencies

**Core Libraries:**
- PyTorch 2.2.2 (CUDA 12.1)
- OpenCV 4.11.0
- scikit-image 0.26.0 (SSIM)
- h5py 3.14.0
- rosbags 0.11.0
- mcap 1.3.1

**Hardware:**
- NVIDIA A100 80GB GPU
- CUDA 13.0 driver support