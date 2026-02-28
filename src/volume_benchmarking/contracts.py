"""Shared contracts for benchmark payloads, cells, and runtime settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VolumeRecord:
    """One volume row from the benchmark catalog."""

    scan_id: str
    nifti_path: str
    modality: str = "UNKNOWN"


@dataclass(frozen=True)
class BenchmarkCellConfig:
    """One cell in the benchmark matrix."""

    backend: str
    cache_state: str
    workers: int
    n_patches: int
    batch_size: int
    warmup_batches: int
    timed_batches: int

    @property
    def cell_id(self) -> str:
        return (
            f"{self.backend}__{self.cache_state}__w{self.workers}"
            f"__n{self.n_patches}__b{self.batch_size}"
        )


@dataclass
class BenchmarkRuntimeConfig:
    """Runtime settings shared across cells."""

    catalog_path: str
    device: str = "auto"
    output_root: str = "."
    pool_size: int = 16
    visits_per_scan: int = 64
    prefetch_replacements: int = 2
    cold_cache_reset_process: bool = True
    enable_nsight: bool = False
    nsight_cmd: str = ""
    seed: int = 42

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root).resolve()


@dataclass
class WorkerMetrics:
    """Per-worker aggregated metrics."""

    pid: int
    peak_rss_gb: float = 0.0
    total_wait_ms: float = 0.0
    wait_samples: int = 0

    @property
    def mean_wait_ms(self) -> float:
        if self.wait_samples <= 0:
            return 0.0
        return self.total_wait_ms / float(self.wait_samples)


@dataclass
class CellMetrics:
    """Final metric bundle for one cell."""

    run_id: str
    cell_id: str
    backend: str
    cache_state: str
    workers: int
    n_patches: int
    batch_size: int
    warmup_batches: int
    timed_batches: int
    time_to_first_batch_ms: float
    throughput_samples_per_sec: float
    timed_wall_seconds: float
    samples_timed: int
    peak_cpu_rss_gb_global: float
    peak_vram_allocated_gb: float
    peak_vram_reserved_gb: float
    gpu_starvation_wait_ms_total: float
    gpu_starvation_wait_ms_mean: float
    replacement_wait_ms_total: float
    worker_metrics: list[WorkerMetrics] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSample:
    """Contract that all dataloader backends must emit per sample."""

    patches_a: Any
    patches_b: Any
    rel_coords_a_mm: Any
    rel_coords_b_mm: Any
    anchor_delta_mm_a_frame: Any
    rotation_b_euler_deg: Any
    rotation_b_matrix: Any
    meta: dict[str, Any]
