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
CSV/CSV.GZ/Parquet with required column:
- `nifti_path`

Optional columns:
- `scan_id`
- `modality`

## Run benchmark matrix
```bash
vb bench run --matrix configs/matrix.default.yaml --runtime configs/runtime.default.yaml --catalog /path/to/catalog.csv --backends all
```

## Run single cell
```bash
vb bench run-cell --backend custom_fused --cache-state warm_pool --workers 4 --n-patches 64 --batch-size 4 --catalog /path/to/catalog.csv
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
