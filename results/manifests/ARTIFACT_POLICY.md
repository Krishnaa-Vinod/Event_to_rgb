# Artifact Commit Policy

This document describes what is committed to the repository vs. what is ignored.

## Committed to Git

- **Code**: All source code in `src/`, `scripts/`, `tests/`
- **Documentation**: All files in `docs/`
- **Configuration**: All files in `configs/`
- **Small summary results**: CSV/JSON files under 1MB in `results/reports/`
- **Representative samples**: A few PNG images showing reconstruction quality
- **Environment specs**: `environment.yml`, `requirements.txt`

## Ignored (Not Committed)

- **Raw datasets**: Original MCAP/bag files, H5 files
- **Model weights**: PyTorch checkpoints and pretrained models
- **Full output directories**: Complete frame reconstruction sets
- **Third-party repositories**: Cloned upstream model repos
- **Cache files**: Python bytecode, pytest cache
- **Large generated artifacts**: Videos, full image sequences

## Size Guidelines

- Committed images: Max 5-10 representative samples
- Committed reports: JSON/CSV files < 1MB
- Repository total: Aim to keep under 100MB

## Rationale

This policy ensures the repository remains lightweight and focused on reproducible code while providing sufficient examples and documentation for users to understand the results.

Large datasets and model weights are expected to be downloaded separately using the provided scripts.