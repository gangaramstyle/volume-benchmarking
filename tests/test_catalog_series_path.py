from __future__ import annotations

import csv
from pathlib import Path

from volume_benchmarking.catalog import load_catalog
from volume_benchmarking.payload import resolve_nifti_path


def test_catalog_accepts_series_path_only(tmp_path: Path) -> None:
    series_dir = tmp_path / "series1"
    series_dir.mkdir()
    catalog = tmp_path / "catalog.csv"

    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scan_id", "series_path", "modality"])
        writer.writeheader()
        writer.writerow({"scan_id": "s1", "series_path": str(series_dir), "modality": "CT"})

    rows = load_catalog(str(catalog))
    assert len(rows) == 1
    assert rows[0].nifti_path == str(series_dir)


def test_resolve_nifti_path_from_series_picks_largest(tmp_path: Path) -> None:
    series_dir = tmp_path / "series2"
    series_dir.mkdir()
    small = series_dir / "small.nii"
    big = series_dir / "big.nii.gz"
    small.write_bytes(b"123")
    big.write_bytes(b"123456789")

    resolved = resolve_nifti_path(str(series_dir))
    assert resolved == str(big)
