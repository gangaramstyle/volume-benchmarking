"""Runtime profiling helpers for benchmark cells."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable


def _try_import_psutil():
    try:
        import psutil
    except Exception:
        return None
    return psutil


def _try_import_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _try_import_nvml():
    try:
        import pynvml
    except Exception:
        return None
    return pynvml


@dataclass
class WorkerSnapshot:
    pid: int
    rss_gb: float


@dataclass
class RuntimeProfiler:
    """Collect memory and wait-time metrics over a benchmark cell run."""

    device: str
    peak_cpu_rss_gb_global: float = 0.0
    peak_vram_allocated_gb: float = 0.0
    peak_vram_reserved_gb: float = 0.0
    gpu_starvation_wait_ms_total: float = 0.0
    gpu_starvation_wait_samples: int = 0
    worker_peak_rss_gb: dict[int, float] = field(default_factory=dict)
    _nvml_device_index: int = 0
    _nvml_initialized: bool = False

    def __post_init__(self) -> None:
        pynvml = _try_import_nvml()
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                self._nvml_initialized = False

    def close(self) -> None:
        if self._nvml_initialized:
            pynvml = _try_import_nvml()
            if pynvml is not None:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            self._nvml_initialized = False

    def record_wait_ms(self, wait_ms: float) -> None:
        self.gpu_starvation_wait_ms_total += float(wait_ms)
        self.gpu_starvation_wait_samples += 1

    def sample_memory(self) -> None:
        psutil = _try_import_psutil()
        if psutil is not None:
            proc = psutil.Process(os.getpid())
            rss_gb = float(proc.memory_info().rss) / (1024.0**3)
            self.peak_cpu_rss_gb_global = max(self.peak_cpu_rss_gb_global, rss_gb)
            self.worker_peak_rss_gb[proc.pid] = max(self.worker_peak_rss_gb.get(proc.pid, 0.0), rss_gb)

            for child in proc.children(recursive=True):
                try:
                    child_rss_gb = float(child.memory_info().rss) / (1024.0**3)
                except Exception:
                    continue
                self.peak_cpu_rss_gb_global = max(self.peak_cpu_rss_gb_global, child_rss_gb)
                self.worker_peak_rss_gb[child.pid] = max(
                    self.worker_peak_rss_gb.get(child.pid, 0.0),
                    child_rss_gb,
                )

        torch = _try_import_torch()
        if torch is not None and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                alloc = float(torch.cuda.max_memory_allocated()) / (1024.0**3)
                reserv = float(torch.cuda.max_memory_reserved()) / (1024.0**3)
                self.peak_vram_allocated_gb = max(self.peak_vram_allocated_gb, alloc)
                self.peak_vram_reserved_gb = max(self.peak_vram_reserved_gb, reserv)
            except Exception:
                pass

            if self._nvml_initialized:
                pynvml = _try_import_nvml()
                if pynvml is not None:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(self._nvml_device_index)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        used = float(mem.used) / (1024.0**3)
                        self.peak_vram_reserved_gb = max(self.peak_vram_reserved_gb, used)
                    except Exception:
                        pass

    def mean_wait_ms(self) -> float:
        if self.gpu_starvation_wait_samples <= 0:
            return 0.0
        return self.gpu_starvation_wait_ms_total / float(self.gpu_starvation_wait_samples)

    def iter_worker_snapshots(self) -> Iterable[WorkerSnapshot]:
        for pid, peak in sorted(self.worker_peak_rss_gb.items()):
            yield WorkerSnapshot(pid=pid, rss_gb=peak)
