from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from volume_benchmarking.backends import make_backend
from volume_benchmarking.backends.base import CellContext
from volume_benchmarking.catalog import load_catalog
from volume_benchmarking.contracts import BenchmarkCellConfig, BenchmarkRuntimeConfig
from volume_benchmarking.rust_bridge import bridge_available


def _write_synthetic_nifti(path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    arr = np.random.randn(16, 16, 8).astype(np.float32)
    img = nib.Nifti1Image(arr, affine=np.eye(4, dtype=np.float32))
    nib.save(img, str(path))


def test_medrs_prepare_bridge_requirement(tmp_path: Path) -> None:
    nifti_path = tmp_path / "sample.nii.gz"
    _write_synthetic_nifti(nifti_path)

    catalog_path = tmp_path / "catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scan_id", "nifti_path", "modality"])
        writer.writeheader()
        writer.writerow({"scan_id": "scan1", "nifti_path": str(nifti_path), "modality": "CT"})

    records = load_catalog(str(catalog_path))
    runtime = BenchmarkRuntimeConfig(catalog_path=str(catalog_path), device="cpu")
    cell = BenchmarkCellConfig(
        backend="medrs",
        cache_state="warm_pool",
        workers=4,
        n_patches=16,
        batch_size=1,
        warmup_batches=1,
        timed_batches=1,
    )
    ctx = CellContext(cell=cell, runtime=runtime, records=records)

    backend = make_backend("medrs", records=records, runtime=runtime)
    if bridge_available():
        backend.prepare(ctx)
    else:
        with pytest.raises(RuntimeError, match="Rust bridge is required"):
            backend.prepare(ctx)
