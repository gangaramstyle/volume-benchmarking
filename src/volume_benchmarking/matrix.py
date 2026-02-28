"""Benchmark matrix loading and cell expansion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from volume_benchmarking.contracts import BenchmarkCellConfig


class MatrixConfigError(RuntimeError):
    pass


def load_matrix_config(path: str) -> dict:
    matrix_path = Path(path).expanduser().resolve()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix config not found: {matrix_path}")
    with matrix_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "matrix" not in data or "run" not in data:
        raise MatrixConfigError("Matrix config must contain top-level keys: matrix, run")
    return data


def matrix_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def expand_cells(config: dict, backends: list[str] | None = None) -> list[BenchmarkCellConfig]:
    mx = config["matrix"]
    run = config["run"]

    cache_states = [str(v) for v in mx.get("cache_states", [])]
    workers = [int(v) for v in mx.get("workers", [])]
    n_patches = [int(v) for v in mx.get("n_patches", [])]
    batch_sizes = [int(v) for v in mx.get("batch_sizes", [])]

    if backends is None:
        backends = [str(v) for v in run.get("backends", [])]

    warmup = int(run.get("warmup_batches", 10))
    timed = int(run.get("timed_batches", 50))

    cells: list[BenchmarkCellConfig] = []
    for backend in backends:
        for cache_state in cache_states:
            for w in workers:
                for n in n_patches:
                    for bs in batch_sizes:
                        cells.append(
                            BenchmarkCellConfig(
                                backend=str(backend),
                                cache_state=str(cache_state),
                                workers=int(w),
                                n_patches=int(n),
                                batch_size=int(bs),
                                warmup_batches=warmup,
                                timed_batches=timed,
                            )
                        )
    return cells
