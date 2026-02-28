"""Catalog loading for nifti_path-based benchmark inputs."""

from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path
from typing import Any

from volume_benchmarking.contracts import VolumeRecord


def _scan_id_for_row(row: dict[str, Any]) -> str:
    explicit = str(row.get("scan_id", "")).strip()
    if explicit:
        return explicit
    nifti_path = str(row.get("nifti_path", "")).strip()
    digest = hashlib.sha1(nifti_path.encode("utf-8")).hexdigest()[:12]
    return f"scan_{digest}"


def _resolve_input_path(row: dict[str, Any]) -> str:
    nifti_path = str(row.get("nifti_path", "")).strip()
    if nifti_path:
        return nifti_path
    series_path = str(row.get("series_path", "")).strip()
    if series_path:
        return series_path
    raise ValueError("Catalog row missing required column: 'nifti_path' or 'series_path'")


def _load_csv_like(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix.endswith("gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import polars as pl  # type: ignore

        return pl.read_parquet(path).to_dicts()
    except Exception:
        try:
            import pandas as pd  # type: ignore

            return pd.read_parquet(path).to_dict("records")
        except Exception as exc:
            raise RuntimeError(
                "Parquet catalog requires polars or pandas+pyarrow"
            ) from exc


def load_catalog(path: str) -> list[VolumeRecord]:
    """Load volume records from csv/csv.gz/parquet catalog.

    Required column: one of `nifti_path` or `series_path`
    Optional columns: `scan_id`, `modality`
    """
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    if catalog_path.suffix == ".parquet":
        rows = _load_parquet(catalog_path)
    else:
        rows = _load_csv_like(catalog_path)

    records: list[VolumeRecord] = []
    for row in rows:
        resolved_path = _resolve_input_path(row)
        records.append(
            VolumeRecord(
                scan_id=_scan_id_for_row(row),
                nifti_path=resolved_path,
                modality=str(row.get("modality", "UNKNOWN") or "UNKNOWN").upper(),
            )
        )
    if not records:
        raise RuntimeError("Catalog loaded successfully but contains zero rows")
    return records
