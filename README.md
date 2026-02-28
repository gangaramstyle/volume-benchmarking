# Volume Benchmarking

Asymmetric 3D-to-2D SSL dataloader benchmark harness comparing:
- `custom_fused`
- `monai`
- `medrs`

## What it benchmarks
Each sample yields two views from the same 3D volume:
- A: native axis-aligned extraction (no 3D rotation interpolation)
- B: rotated extraction with continuous 3D transform

Payload contract:
- `patches_a`, `patches_b`: `(N, 1, 16, 16)`
- `rel_coords_a_mm`, `rel_coords_b_mm`: `(N, 3)`
- `anchor_delta_mm_a_frame`: `(3,)`
- `rotation_b_euler_deg`: `(3,)`
- `rotation_b_matrix`: `(3, 3)`

## Install
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
# Optional extras
uv pip install -e .[monai,medrs,test]
```

## Catalog format
CSV/CSV.GZ/Parquet with at least one of:
- `nifti_path` (direct file path), or
- `series_path` (directory containing `.nii`/`.nii.gz`)

Optional columns:
- `scan_id`
- `modality`

Small committed smoke catalog:
- `data/catalog_smoke_32.csv`

Current default filtering behavior:
- keep only `CT` and `MR` modalities
- skip likely non-diagnostic/derived series (for example scout/localizer/reformat-style series)

## Run benchmark matrix
```bash
vb bench run --matrix configs/matrix.default.yaml --runtime configs/runtime.default.yaml --catalog /path/to/catalog.csv --backends all
```

Smaller presets for faster iteration:
```bash
# Quick backend health check (3 cells total)
vb bench run --matrix configs/matrix.smoke.yaml --runtime configs/runtime.default.yaml --catalog data/catalog_smoke_32.csv --backends all

# Fast comparison pass (24 cells total)
vb bench run --matrix configs/matrix.fast.yaml --runtime configs/runtime.default.yaml --catalog data/catalog_smoke_32.csv --backends all
```

## Run single cell
```bash
vb bench run-cell --backend custom_fused --cache-state warm_pool --workers 4 --n-patches 64 --batch-size 4 --catalog /path/to/catalog.csv
```

## Single-file sanity check (descriptive diagnostics)
```bash
vb bench sanity --backend custom_fused --catalog data/catalog_smoke_32.csv --record-index 0 --n-patches 16
vb bench sanity --backend monai --catalog data/catalog_smoke_32.csv --record-index 0 --n-patches 16
vb bench sanity --backend medrs --catalog data/catalog_smoke_32.csv --record-index 0 --n-patches 16
```

## Summarize existing raw runs
```bash
vb bench summarize --input results/raw/*.jsonl
```

## Cluster workflow (single-line)
```bash
cd ~/project_name && vb bench run --matrix configs/matrix.default.yaml --runtime configs/runtime.default.yaml --catalog /path/to/catalog.csv --backends all && git add results/ logs/ && git commit -m "benchmark run $(date +%F_%H%M)" && git push
```

## Local analysis after cluster run
```bash
git pull
vb bench summarize --input results/raw/*.jsonl
```

## Notes
- Cold cache mode is user-space approximation (`cold_approx`) because OS page-cache drop requires root.
- Nsight mode is optional (`--nsight`) and disabled by default.
- DLProf is intentionally not used.
- MedRS uses internal threaded workers; torch `DataLoader` process workers are disabled for this backend to avoid fork deadlocks.
- Runtime safety defaults in `configs/runtime.default.yaml`:
  - `data_loader_timeout_s`: DataLoader worker timeout per fetch (prevents indefinite hangs)
  - `continue_on_cell_error`: continue matrix run after per-cell failures and record `cell_error` events
