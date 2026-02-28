"""Benchmark execution runner, profiling, and report generation."""

from __future__ import annotations

import csv
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from volume_benchmarking.backends import make_backend
from volume_benchmarking.backends.base import CellContext
from volume_benchmarking.catalog import load_catalog
from volume_benchmarking.contracts import BenchmarkCellConfig, BenchmarkRuntimeConfig, CellMetrics, WorkerMetrics
from volume_benchmarking.matrix import expand_cells, load_matrix_config, matrix_hash
from volume_benchmarking.payload import load_nifti_context, resolve_nifti_path
from volume_benchmarking.profiling import RuntimeProfiler


def _now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_sha(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_runtime_config(path: str | None, catalog_override: str | None = None) -> BenchmarkRuntimeConfig:
    payload: dict[str, Any] = {}
    if path:
        p = Path(path).expanduser().resolve()
        with p.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    runtime = payload.get("runtime", {})
    catalog_path = str(catalog_override or runtime.get("catalog_path", "")).strip()
    if not catalog_path:
        raise RuntimeError("Catalog path is required. Pass --catalog or set runtime.catalog_path")

    cfg = BenchmarkRuntimeConfig(
        catalog_path=catalog_path,
        device=str(runtime.get("device", "auto")),
        output_root=str(runtime.get("output_root", ".")),
        pool_size=int(runtime.get("pool_size", 16)),
        visits_per_scan=int(runtime.get("visits_per_scan", 64)),
        prefetch_replacements=int(runtime.get("prefetch_replacements", 2)),
        cold_cache_reset_process=bool(runtime.get("cold_cache_reset_process", True)),
        data_loader_timeout_s=int(runtime.get("data_loader_timeout_s", 180)),
        continue_on_cell_error=bool(runtime.get("continue_on_cell_error", True)),
        enable_nsight=bool(runtime.get("enable_nsight", False)),
        nsight_cmd=str(runtime.get("nsight_cmd", "")),
        seed=int(runtime.get("seed", 42)),
    )
    cfg.device = _resolve_device(cfg.device)
    return cfg


def _ensure_dirs(output_root: Path) -> None:
    (output_root / "results" / "raw").mkdir(parents=True, exist_ok=True)
    (output_root / "results" / "summaries").mkdir(parents=True, exist_ok=True)
    (output_root / "results" / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "logs" / "benchmarks").mkdir(parents=True, exist_ok=True)


def _import_torch_runtime():
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("torch is required to run benchmarks") from exc
    return torch, DataLoader


def _stack_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _ = _import_torch_runtime()
    out: dict[str, Any] = {
        "patches_a": torch.stack([s["patches_a"] for s in batch], dim=0),
        "patches_b": torch.stack([s["patches_b"] for s in batch], dim=0),
        "rel_coords_a_mm": torch.stack([s["rel_coords_a_mm"] for s in batch], dim=0),
        "rel_coords_b_mm": torch.stack([s["rel_coords_b_mm"] for s in batch], dim=0),
        "anchor_delta_mm_a_frame": torch.stack([s["anchor_delta_mm_a_frame"] for s in batch], dim=0),
        "rotation_b_euler_deg": torch.stack([s["rotation_b_euler_deg"] for s in batch], dim=0),
        "rotation_b_matrix": torch.stack([s["rotation_b_matrix"] for s in batch], dim=0),
        "meta": [s["meta"] for s in batch],
    }
    out["replacement_wait_ms_delta_sum"] = float(
        sum(float(m.get("replacement_wait_ms_delta", 0.0)) for m in out["meta"])
    )
    return out


def _move_batch_tensors_to_device(batch: dict[str, Any], device: str) -> None:
    torch, _ = _import_torch_runtime()
    if device == "cpu":
        return
    for key in (
        "patches_a",
        "patches_b",
        "rel_coords_a_mm",
        "rel_coords_b_mm",
        "anchor_delta_mm_a_frame",
        "rotation_b_euler_deg",
        "rotation_b_matrix",
    ):
        if key in batch and hasattr(batch[key], "to"):
            batch[key] = batch[key].to(device, non_blocking=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _write_jsonl_line(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()


def _cell_to_row_dict(metrics: CellMetrics) -> dict[str, Any]:
    return {
        "run_id": metrics.run_id,
        "cell_id": metrics.cell_id,
        "backend": metrics.backend,
        "cache_state": metrics.cache_state,
        "workers": metrics.workers,
        "n_patches": metrics.n_patches,
        "batch_size": metrics.batch_size,
        "warmup_batches": metrics.warmup_batches,
        "timed_batches": metrics.timed_batches,
        "time_to_first_batch_ms": metrics.time_to_first_batch_ms,
        "throughput_samples_per_sec": metrics.throughput_samples_per_sec,
        "timed_wall_seconds": metrics.timed_wall_seconds,
        "samples_timed": metrics.samples_timed,
        "peak_cpu_rss_gb_global": metrics.peak_cpu_rss_gb_global,
        "peak_vram_allocated_gb": metrics.peak_vram_allocated_gb,
        "peak_vram_reserved_gb": metrics.peak_vram_reserved_gb,
        "gpu_starvation_wait_ms_total": metrics.gpu_starvation_wait_ms_total,
        "gpu_starvation_wait_ms_mean": metrics.gpu_starvation_wait_ms_mean,
        "replacement_wait_ms_total": metrics.replacement_wait_ms_total,
    }


def run_single_cell(
    *,
    run_id: str,
    cell: BenchmarkCellConfig,
    runtime: BenchmarkRuntimeConfig,
    records,
    raw_handle,
) -> CellMetrics:
    torch, DataLoader = _import_torch_runtime()
    backend = make_backend(cell.backend, records=records, runtime=runtime)
    ctx = CellContext(cell=cell, runtime=runtime, records=records)
    backend.prepare(ctx)

    dataset = backend.build_dataset(ctx)
    effective_loader_workers = int(cell.workers)
    if cell.backend == "medrs":
        # MedRS parallelism is handled inside the backend to avoid torch process deadlocks.
        effective_loader_workers = 0

    loader_kwargs: dict[str, Any] = {
        "batch_size": cell.batch_size,
        "num_workers": effective_loader_workers,
        "collate_fn": _stack_collate,
        "pin_memory": runtime.device.startswith("cuda"),
    }
    if effective_loader_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2
        if int(runtime.data_loader_timeout_s) > 0:
            loader_kwargs["timeout"] = int(runtime.data_loader_timeout_s)

    loader = DataLoader(dataset, **loader_kwargs)
    profiler = RuntimeProfiler(device=runtime.device)

    data_iter = iter(loader)
    replacement_wait_ms_total = 0.0
    samples_timed = 0

    ttfb_start = time.perf_counter()
    fetch_start = time.perf_counter()
    first_batch = next(data_iter)
    first_wait_ms = (time.perf_counter() - fetch_start) * 1000.0
    time_to_first_batch_ms = (time.perf_counter() - ttfb_start) * 1000.0
    profiler.record_wait_ms(first_wait_ms)
    profiler.sample_memory()
    replacement_wait_ms_total += float(first_batch.get("replacement_wait_ms_delta_sum", 0.0))

    _write_jsonl_line(
        raw_handle,
        {
            "event": "first_batch",
            "run_id": run_id,
            "cell_id": cell.cell_id,
            "backend": cell.backend,
            "cache_state": cell.cache_state,
            "wait_ms": first_wait_ms,
            "time_to_first_batch_ms": time_to_first_batch_ms,
        },
    )

    for _ in range(cell.warmup_batches):
        t0 = time.perf_counter()
        batch = next(data_iter)
        wait_ms = (time.perf_counter() - t0) * 1000.0
        profiler.record_wait_ms(wait_ms)
        profiler.sample_memory()
        replacement_wait_ms_total += float(batch.get("replacement_wait_ms_delta_sum", 0.0))
        _move_batch_tensors_to_device(batch, runtime.device)

    timed_wall_start = time.perf_counter()
    for step_idx in range(cell.timed_batches):
        t0 = time.perf_counter()
        batch = next(data_iter)
        wait_ms = (time.perf_counter() - t0) * 1000.0
        profiler.record_wait_ms(wait_ms)

        replacement_wait_ms_total += float(batch.get("replacement_wait_ms_delta_sum", 0.0))
        _move_batch_tensors_to_device(batch, runtime.device)

        bs = int(batch["patches_a"].shape[0])
        samples_timed += bs
        profiler.sample_memory()

        _write_jsonl_line(
            raw_handle,
            {
                "event": "timed_batch",
                "run_id": run_id,
                "cell_id": cell.cell_id,
                "backend": cell.backend,
                "cache_state": cell.cache_state,
                "step": int(step_idx),
                "batch_size": bs,
                "wait_ms": wait_ms,
                "replacement_wait_ms_delta": float(batch.get("replacement_wait_ms_delta_sum", 0.0)),
            },
        )

    timed_wall_seconds = max(time.perf_counter() - timed_wall_start, 1e-9)
    throughput_samples_per_sec = float(samples_timed) / timed_wall_seconds

    worker_metrics: list[WorkerMetrics] = [
        WorkerMetrics(pid=snap.pid, peak_rss_gb=snap.rss_gb) for snap in profiler.iter_worker_snapshots()
    ]

    result = CellMetrics(
        run_id=run_id,
        cell_id=cell.cell_id,
        backend=cell.backend,
        cache_state=cell.cache_state,
        workers=cell.workers,
        n_patches=cell.n_patches,
        batch_size=cell.batch_size,
        warmup_batches=cell.warmup_batches,
        timed_batches=cell.timed_batches,
        time_to_first_batch_ms=time_to_first_batch_ms,
        throughput_samples_per_sec=throughput_samples_per_sec,
        timed_wall_seconds=timed_wall_seconds,
        samples_timed=samples_timed,
        peak_cpu_rss_gb_global=profiler.peak_cpu_rss_gb_global,
        peak_vram_allocated_gb=profiler.peak_vram_allocated_gb,
        peak_vram_reserved_gb=profiler.peak_vram_reserved_gb,
        gpu_starvation_wait_ms_total=profiler.gpu_starvation_wait_ms_total,
        gpu_starvation_wait_ms_mean=profiler.mean_wait_ms(),
        replacement_wait_ms_total=replacement_wait_ms_total,
        worker_metrics=worker_metrics,
        extra={
            "device": runtime.device,
            "nsight_enabled": runtime.enable_nsight,
            "effective_loader_workers": effective_loader_workers,
        },
    )

    _write_jsonl_line(
        raw_handle,
        {
            "event": "cell_summary",
            **_cell_to_row_dict(result),
            "worker_metrics": [
                {"pid": wm.pid, "peak_rss_gb": wm.peak_rss_gb, "mean_wait_ms": wm.mean_wait_ms}
                for wm in worker_metrics
            ],
            "extra": result.extra,
        },
    )

    profiler.close()
    backend.teardown()
    return result


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]], run_id: str) -> None:
    lines = [f"# Benchmark Summary {run_id}", "", "## Top Throughput Cells", ""]
    ranked = sorted(rows, key=lambda r: float(r["throughput_samples_per_sec"]), reverse=True)
    lines.append("| rank | cell_id | throughput (samples/sec) | ttfb (ms) | peak_cpu_rss_gb | peak_vram_reserved_gb |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for idx, row in enumerate(ranked[:20], start=1):
        lines.append(
            "| {rank} | {cell} | {thr:.3f} | {ttfb:.2f} | {cpu:.3f} | {vram:.3f} |".format(
                rank=idx,
                cell=row["cell_id"],
                thr=float(row["throughput_samples_per_sec"]),
                ttfb=float(row["time_to_first_batch_ms"]),
                cpu=float(row["peak_cpu_rss_gb_global"]),
                vram=float(row["peak_vram_reserved_gb"]),
            )
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Cold cache mode is user-space approximation (`cold_approx`).",
            "- MONAI and MedRS are strict paths (no hidden fallback extraction path).",
            "- MedRS uses internal threaded workers and torch DataLoader process workers are disabled.",
            "- Nsight tracing is optional and disabled unless explicitly enabled.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_record(records, record_index: int, scan_id: str | None):
    if scan_id:
        for rec in records:
            if str(rec.scan_id) == str(scan_id):
                return rec
        raise RuntimeError(f"No record found with scan_id='{scan_id}'")
    if not records:
        raise RuntimeError("Catalog is empty")
    idx = int(record_index) % len(records)
    return records[idx]


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(v) for v in shape]
    except Exception:
        return None


def _validate_sample_contract(sample: dict[str, Any], n_patches: int) -> dict[str, Any]:
    required = [
        "patches_a",
        "patches_b",
        "rel_coords_a_mm",
        "rel_coords_b_mm",
        "anchor_delta_mm_a_frame",
        "rotation_b_euler_deg",
        "rotation_b_matrix",
        "meta",
    ]
    missing = [k for k in required if k not in sample]
    checks: dict[str, Any] = {
        "missing_keys": missing,
        "has_all_required_keys": len(missing) == 0,
        "shapes": {k: _shape_of(sample.get(k)) for k in required if k != "meta"},
    }
    expected = {
        "patches_a": [int(n_patches), 1, 16, 16],
        "patches_b": [int(n_patches), 1, 16, 16],
        "rel_coords_a_mm": [int(n_patches), 3],
        "rel_coords_b_mm": [int(n_patches), 3],
        "anchor_delta_mm_a_frame": [3],
        "rotation_b_euler_deg": [3],
        "rotation_b_matrix": [3, 3],
    }
    checks["shape_matches"] = {
        k: checks["shapes"].get(k) == v for k, v in expected.items()
    }
    checks["all_shapes_match"] = all(checks["shape_matches"].values())
    return checks


def summarize_raw_metrics(input_patterns: list[str], output_root: str = ".") -> tuple[Path, Path]:
    files: list[Path] = []
    for pat in input_patterns:
        files.extend(Path(p) for p in glob.glob(pat))
    files = sorted({p.resolve() for p in files})
    if not files:
        raise RuntimeError("No raw metric files matched input patterns")

    rows: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("event") == "cell_summary":
                    rows.append({
                        k: payload[k]
                        for k in [
                            "run_id",
                            "cell_id",
                            "backend",
                            "cache_state",
                            "workers",
                            "n_patches",
                            "batch_size",
                            "warmup_batches",
                            "timed_batches",
                            "time_to_first_batch_ms",
                            "throughput_samples_per_sec",
                            "timed_wall_seconds",
                            "samples_timed",
                            "peak_cpu_rss_gb_global",
                            "peak_vram_allocated_gb",
                            "peak_vram_reserved_gb",
                            "gpu_starvation_wait_ms_total",
                            "gpu_starvation_wait_ms_mean",
                            "replacement_wait_ms_total",
                        ]
                        if k in payload
                    })

    if not rows:
        raise RuntimeError("No cell_summary events found in provided raw files")

    output = Path(output_root).resolve()
    _ensure_dirs(output)
    run_id = f"summary_{_now_run_id()}"
    csv_path = output / "results" / "summaries" / f"{run_id}.csv"
    md_path = output / "results" / "summaries" / f"{run_id}.md"
    _write_summary_csv(csv_path, rows)
    _write_summary_markdown(md_path, rows, run_id=run_id)
    return csv_path, md_path


def run_sanity_check(
    *,
    backend: str,
    catalog: str,
    runtime_path: str | None,
    record_index: int,
    scan_id: str | None,
    n_patches: int,
    cache_state: str,
) -> dict[str, Any]:
    runtime = _load_runtime_config(runtime_path, catalog_override=catalog)
    records = load_catalog(runtime.catalog_path)
    rec = _select_record(records, record_index=record_index, scan_id=scan_id)

    resolved_path = resolve_nifti_path(rec.nifti_path)
    ctx = load_nifti_context(scan_id=rec.scan_id, nifti_path=rec.nifti_path)

    cell = BenchmarkCellConfig(
        backend=backend,
        cache_state=cache_state,
        workers=0,
        n_patches=int(n_patches),
        batch_size=1,
        warmup_batches=0,
        timed_batches=1,
    )
    backend_impl = make_backend(cell.backend, records=[rec], runtime=runtime)
    cell_ctx = CellContext(cell=cell, runtime=runtime, records=[rec])
    backend_impl.prepare(cell_ctx)

    sample_start = time.perf_counter()
    dataset = backend_impl.build_dataset(cell_ctx)
    sample = next(iter(dataset))
    sample_elapsed_ms = (time.perf_counter() - sample_start) * 1000.0
    backend_impl.teardown()

    contract = _validate_sample_contract(sample, n_patches=n_patches)
    meta = sample.get("meta", {}) if isinstance(sample, dict) else {}

    return {
        "backend": backend,
        "catalog_path": runtime.catalog_path,
        "record_index": int(record_index),
        "selected_scan_id": str(rec.scan_id),
        "selected_modality": str(rec.modality),
        "input_path": str(rec.nifti_path),
        "resolved_nifti_path": str(resolved_path),
        "volume": {
            "shape_xyz": [int(v) for v in ctx.volume_xyz.shape],
            "spacing_mm": [float(v) for v in ctx.spacing_mm.tolist()],
            "dtype": str(ctx.volume_xyz.dtype),
            "min": float(ctx.volume_xyz.min()),
            "max": float(ctx.volume_xyz.max()),
            "mean": float(ctx.volume_xyz.mean()),
        },
        "runtime_device": runtime.device,
        "sample_generation_ms": float(sample_elapsed_ms),
        "contract": contract,
        "sample_meta": meta,
    }


def run_matrix_benchmark(
    *,
    matrix_path: str,
    runtime_path: str | None,
    backends: list[str] | None,
    catalog_override: str | None = None,
) -> dict[str, Path | str]:
    matrix_cfg = load_matrix_config(matrix_path)
    runtime = _load_runtime_config(runtime_path, catalog_override=catalog_override)
    runtime.seed = int(matrix_cfg.get("run", {}).get("seed", runtime.seed))

    if backends is None:
        backends = [str(v) for v in matrix_cfg.get("run", {}).get("backends", [])]
    if not backends:
        raise RuntimeError("No backends selected")

    records = load_catalog(runtime.catalog_path)
    cells = expand_cells(matrix_cfg, backends=backends)

    output_root = runtime.output_root_path
    _ensure_dirs(output_root)

    run_id = _now_run_id()
    raw_path = output_root / "results" / "raw" / f"{run_id}.jsonl"

    rows: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for cell in cells:
            try:
                metrics = run_single_cell(
                    run_id=run_id,
                    cell=cell,
                    runtime=runtime,
                    records=records,
                    raw_handle=raw_handle,
                )
                rows.append(_cell_to_row_dict(metrics))
            except Exception as exc:
                _write_jsonl_line(
                    raw_handle,
                    {
                        "event": "cell_error",
                        "run_id": run_id,
                        "cell_id": cell.cell_id,
                        "backend": cell.backend,
                        "cache_state": cell.cache_state,
                        "workers": cell.workers,
                        "n_patches": cell.n_patches,
                        "batch_size": cell.batch_size,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                if not runtime.continue_on_cell_error:
                    raise

    csv_path = output_root / "results" / "summaries" / f"{run_id}.csv"
    md_path = output_root / "results" / "summaries" / f"{run_id}.md"
    manifest_path = output_root / "results" / "manifests" / f"{run_id}.json"

    _write_summary_csv(csv_path, rows)
    _write_summary_markdown(md_path, rows, run_id=run_id)

    manifest = {
        "run_id": run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(output_root),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": runtime.device,
        "catalog_path": runtime.catalog_path,
        "matrix_path": str(Path(matrix_path).resolve()),
        "matrix_hash": matrix_hash(matrix_cfg),
        "backends": backends,
        "cells": len(cells),
        "cold_cache_mode": "approx_user_space",
        "nsight_enabled": runtime.enable_nsight,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "raw": raw_path,
        "summary_csv": csv_path,
        "summary_md": md_path,
        "manifest": manifest_path,
    }


def run_cell_from_cli(
    *,
    backend: str,
    cache_state: str,
    workers: int,
    n_patches: int,
    batch_size: int,
    catalog: str,
    runtime_path: str | None,
    warmup_batches: int,
    timed_batches: int,
) -> dict[str, Path | str]:
    runtime = _load_runtime_config(runtime_path, catalog_override=catalog)
    records = load_catalog(runtime.catalog_path)
    output_root = runtime.output_root_path
    _ensure_dirs(output_root)

    run_id = _now_run_id()
    raw_path = output_root / "results" / "raw" / f"{run_id}.jsonl"

    cell = BenchmarkCellConfig(
        backend=backend,
        cache_state=cache_state,
        workers=workers,
        n_patches=n_patches,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        timed_batches=timed_batches,
    )

    with raw_path.open("w", encoding="utf-8") as raw_handle:
        metrics = run_single_cell(
            run_id=run_id,
            cell=cell,
            runtime=runtime,
            records=records,
            raw_handle=raw_handle,
        )

    row = _cell_to_row_dict(metrics)
    csv_path = output_root / "results" / "summaries" / f"{run_id}.csv"
    md_path = output_root / "results" / "summaries" / f"{run_id}.md"
    _write_summary_csv(csv_path, [row])
    _write_summary_markdown(md_path, [row], run_id=run_id)

    return {
        "run_id": run_id,
        "raw": raw_path,
        "summary_csv": csv_path,
        "summary_md": md_path,
    }
