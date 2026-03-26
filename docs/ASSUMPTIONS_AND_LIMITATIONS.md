# Assumptions and Limitations

This document explicitly describes the assumptions made in the pipeline implementation and known limitations of the current approach.

## Core Assumptions

### **1. Reconstruction Output Format**
**Assumption**: E2VID and FireNet produce **grayscale intensity reconstructions**, not true RGB.

**Rationale**:
- Standard event cameras (DVS/DAVIS) are monochrome sensors
- E2VID/FireNet are trained on grayscale intensity data
- No evidence of color event reconstruction in the model architectures

**Implication**: All quantitative comparisons use grayscale-converted reference images.

### **2. Temporal Alignment Policy**
**Assumption**: ±50ms temporal matching window is sufficient for fair comparison.

**Rationale**:
- H5 voxel windows are 250ms, providing inherent temporal tolerance
- Event processing introduces variable latency
- Human visual perception tolerance for motion alignment

**Alternative approaches considered**: ECC-based spatial alignment was deemed more complex than necessary.

### **4. H5-Direct Route Fidelity**
**CRITICAL LIMITATION**: The H5-direct reconstruction route is **approximate** and not equivalent to bag-direct.

**Technical details**:
- H5 files contain pre-computed 5-bin voxel grids (5×720×1280)
- H5-direct converts voxel bins back to pseudo-events using fixed thresholds:
  - Positive events: voxel_value > 0.1
  - Negative events: voxel_value < -0.1
- Temporal spread: 5 bins distributed over 250ms window
- This pseudo-event reconstruction may not match the original event stream

**Implications**:
- H5-direct results are labeled `route_fidelity: "approximate_voxel_to_pseudo_event"`
- H5-direct is reported separately from the main fair leaderboard
- Only bag-direct routes provide `route_fidelity: "exact_raw_event"`

**Use cases**: H5-direct is useful for development and debugging but should not be used for authoritative performance claims.

### **5. Image Geometry Handling**
**Assumption**: Center-crop/resize to common 720×1280 resolution is fair.

**Geometry differences**:
- Event data: 720×1280
- H5 RGB references: 1024×1280
- Bag RGB images: bayer_rggb8 format

**Preprocessing policy**:
1. Convert all inputs to grayscale
2. Resize to 720×1280 using bilinear interpolation
3. Normalize to uint8 [0, 255] range

### **4. H5 Voxel Compatibility**
**Critical Assumption**: H5 voxels can be approximately converted back to events for E2VID processing.

**Process**:
```python
# Threshold-based event generation
pos_events = voxel > 0.1   # Positive events
neg_events = voxel < -0.1  # Negative events
```

**Limitation**: This conversion is **approximate** and may not match raw bag→E2VID processing exactly.

### **5. Fair Comparison Protocol**
**Assumption**: 250ms fixed-duration windows provide fair comparison across all methods.

**Justification**:
- Matches H5 pipeline voxel window duration
- Sufficient temporal support for reconstruction quality
- Eliminates window-size confounding factors

**Secondary comparison**: Bag-only methods tested with multiple window sizes [33ms, 50ms, 100ms, 250ms] separately.

## Known Limitations

### **1. H5 Direct Route Accuracy**

**Issue**: Voxel→event conversion is not numerically identical to bag→voxel→E2VID pipeline.

**Impact**: H5-direct results should be labeled as "approximate" rather than exact equivalents.

**Root cause**:
- Different voxelization parameters between source pipeline and E2VID
- Loss of fine temporal information in voxel representation
- Threshold-based event regeneration introduces artifacts

**Mitigation**: Clearly document this limitation in all result presentations.

### **2. RGB Reference Format Support**

**Limitation**: Bag RGB images are in `bayer_rggb8` format, not directly supported.

**Current status**: H5 RGB references used exclusively for evaluation.

**Impact**:
- Cannot validate bag temporal synchronization against bag RGB
- Assumes H5 RGB accurately represents original camera output

**Future work**: Implement bayer pattern decoding for comprehensive validation.

### **3. Timestamp Coordinate Systems**

**Issue**: Mismatched timestamp representations between data sources.

**Details**:
- Events: Relative timestamps starting from 0.0s
- H5 data: Absolute timestamps (~1.77e12 nanoseconds)
- Evaluation: Requires additional timestamp normalization

**Workaround**: Convert to common relative coordinate system before matching.

### **4. FireNet Model Availability**

**Limitation**: FireNet weights require manual download from Google Drive.

**Impact**: Automated pipeline setup cannot fetch FireNet weights automatically.

**Status**: E2VID fully functional, FireNet repository cloned but weights pending.

### **5. Color Reconstruction**

**Limitation**: Pipeline focuses on grayscale intensity reconstruction only.

**Justification**:
- Standard event cameras are monochrome
- No evidence of color event data in provided datasets
- E2VID/FireNet models trained on grayscale data

**Extension needed**: Color event camera support would require different models and preprocessing.

### **6. Evaluation Completeness**

**Missing metrics**:
- LPIPS perceptual similarity (installation complexity)
- Dynamic range metrics (HDR capability assessment)
- Temporal consistency metrics (inter-frame similarity)

**Current metrics**: MSE, MAE, PSNR, SSIM provide comprehensive spatial quality assessment.

## Fairness Considerations

### **1. Window Duration Impact**

**Concern**: Fixed 250ms windows may favor methods optimized for this duration.

**Mitigation**: Secondary bag-only comparison explores multiple window sizes.

**Transparency**: All window durations clearly documented in results.

### **2. Preprocessing Normalization**

**Concern**: Image normalization may mask method-specific dynamic range differences.

**Approach**: Raw intensity statistics reported alongside normalized metrics.

**Alternative**: Could compute metrics on unnormalized intensity ranges.

### **3. Temporal Matching Bias**

**Concern**: ±50ms matching window may favor methods with specific temporal characteristics.

**Documentation**: All timestamp deltas recorded and reported.

**Sensitivity analysis**: Could test multiple matching window sizes.

## Data Quality Assumptions

### **1. H5 Pipeline Correctness**

**Assumption**: Source `eventnavpp_datagen_pipeline` produces correct H5 voxelizations.

**Validation**: Spot checks on voxel statistics and temporal alignment performed.

**Risk**: Errors in source pipeline would propagate to H5-direct route evaluation.

### **2. Event Data Integrity**

**Assumption**: EVT3 decoding preserves event temporal and spatial accuracy.

**Validation**: Event coordinate bounds checking and temporal monotonicity verified.

**Confidence**: Source pipeline utilities extensively tested in original project.

### **3. Model Weight Authenticity**

**Assumption**: E2VID weights from official source (`rpg.ifi.uzh.ch`) are correct and uncorrupted.

**Verification**: SHA256 checksum validation recommended for production use.

## Reproducibility Guarantees

### **✅ Guaranteed Reproducible**
- Environment setup (conda environment exported)
- Data paths and configurations (example configs provided)
- Model weights and versions (URLs and checksums documented)
- Processing scripts and parameters (all code committed)

### **⚠️ Partially Reproducible**
- Hardware-specific CUDA performance characteristics
- Random initialization in model inference (if any)
- Filesystem timing dependencies

### **❌ Not Reproducible**
- Manual FireNet weight download step
- Source data modification timestamps
- Interactive debugging and development sessions

## Conclusion

This pipeline implementation prioritizes:
1. **Scientific rigor** in fair comparison methodology
2. **Transparency** about limitations and approximations
3. **Practical utility** for event camera reconstruction research
4. **Reproducibility** through comprehensive documentation

All limitations are explicitly acknowledged rather than hidden, enabling users to make informed decisions about result interpretation and potential improvements.