# Event-to-RGB Reconstruction Gallery

## Method Comparison: E2VID Results

### Bag-Direct Route (Exact)
- **Path:** `outputs/targeted_benchmark/e2vid_bag_direct/e2vid_window_250ms/reconstruction/`
- **Frames:** 8 images generated
- **Quality:** Mean intensity 117.0 ± 17.0, no blank frames
- **Route Fidelity:** EXACT - Direct from bag events

**Sample frames:**
- `frame_0000885867.png`
- `frame_0001317041.png`
- `frame_0009097485.png`
- `frame_0016881821.png`
- `frame_0025472095.png`
- `frame_0032720431.png`
- `frame_0039984234.png`
- `frame_0045594791.png`

### H5-Direct Route (Approximate)
- **Path:** `outputs/targeted_benchmark/e2vid_h5_direct/`
- **Frames:** 8 images generated
- **Quality:** Mean intensity 175.8 ± 3.6, no blank frames
- **Route Fidelity:** APPROXIMATE - H5 voxel pseudo-events

**Sample frames:** 8 reconstruction outputs available

## Key Observations

### Intensity Characteristics
- **Bag-Direct:** Higher variance (std=17.0) suggests more natural event dynamics
- **H5-Direct:** Lower variance (std=3.6) indicates voxel quantization smoothing
- **H5-Direct:** Higher mean intensity (175.8 vs 117.0) due to pseudo-event processing

### Reconstruction Success
✓ Both methods successfully generate valid, non-blank images
✓ Consistent frame counts demonstrate pipeline reliability
✓ All outputs pass sanity validation checks

### Route Comparison
- **Exact Route:** Preserves original event timing and dynamics
- **Approximate Route:** Introduces voxel quantization artifacts but maintains reconstruction capability

## Generated Artifacts

All reconstruction images, manifests, and evaluation reports are available in:
- `outputs/targeted_benchmark/`
- `results/reports/`

## Usage

To view reconstructions:
```bash
# View bag-direct E2VID results
ls outputs/targeted_benchmark/e2vid_bag_direct/e2vid_window_250ms/reconstruction/

# View H5-direct E2VID results
ls outputs/targeted_benchmark/e2vid_h5_direct/
```

---
*Event-to-RGB Pipeline Benchmarking Results*
*Generated: 2026-03-26*