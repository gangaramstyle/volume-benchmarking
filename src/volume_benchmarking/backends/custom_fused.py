"""Custom fused backend using nibabel I/O, warm-pool caching, and torch grid_sample."""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from volume_benchmarking.backends.base import BackendAdapter, CellContext
from volume_benchmarking.payload import VolumeContext, build_asymmetric_sample, load_nifti_context

try:
    from torch.utils.data import IterableDataset
except Exception:  # pragma: no cover - guarded at runtime
    class IterableDataset:  # type: ignore
        pass


@dataclass
class _PoolSlot:
    record_idx: int
    ctx: VolumeContext
    visits: int = 0
    replacing: bool = False
    future: Future | None = None
    future_started: float = 0.0


class _WarmPool:
    def __init__(
        self,
        *,
        records,
        pool_size: int,
        visits_per_scan: int,
        prefetch_replacements: int,
        seed: int,
    ) -> None:
        self._records = list(records)
        self._pool_size = min(max(1, int(pool_size)), len(self._records))
        self._visits_per_scan = max(1, int(visits_per_scan))
        self._rng = random.Random(int(seed))
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(prefetch_replacements)))
        self._slots: list[_PoolSlot] = []
        self._rr_index = 0
        self._next_record_cursor = 0
        self._lock = threading.Lock()
        self._replacement_wait_ms_delta = 0.0

    def _next_record(self):
        rec = self._records[self._next_record_cursor % len(self._records)]
        self._next_record_cursor += 1
        return rec

    def _load_record(self, record) -> VolumeContext:
        return load_nifti_context(scan_id=record.scan_id, nifti_path=record.nifti_path)

    def bootstrap(self) -> None:
        attempts = 0
        while len(self._slots) < self._pool_size and attempts < max(len(self._records) * 2, self._pool_size):
            attempts += 1
            rec = self._next_record()
            try:
                ctx = self._load_record(rec)
            except Exception:
                continue
            self._slots.append(_PoolSlot(record_idx=(self._next_record_cursor - 1), ctx=ctx))
        if not self._slots:
            raise RuntimeError("WarmPool bootstrap failed: no valid volume could be loaded")

    def poll_replacements(self) -> float:
        delta = 0.0
        with self._lock:
            for slot in self._slots:
                if slot.future is None or not slot.future.done():
                    continue
                wait_ms = max((time.perf_counter() - slot.future_started) * 1000.0, 0.0)
                delta += wait_ms
                try:
                    slot.ctx = slot.future.result()
                    slot.visits = 0
                except Exception:
                    pass
                slot.future = None
                slot.future_started = 0.0
                slot.replacing = False
            self._replacement_wait_ms_delta += delta

        out = self._replacement_wait_ms_delta
        self._replacement_wait_ms_delta = 0.0
        return out

    def _maybe_schedule_replacement(self, slot: _PoolSlot) -> None:
        if slot.replacing or slot.visits < self._visits_per_scan:
            return
        rec = self._next_record()
        slot.replacing = True
        slot.future_started = time.perf_counter()
        slot.future = self._executor.submit(self._load_record, rec)

    def sample(self) -> tuple[VolumeContext, float]:
        if not self._slots:
            raise RuntimeError("WarmPool has no slots")
        delta_wait = self.poll_replacements()

        idx = self._rr_index % len(self._slots)
        self._rr_index += 1
        slot = self._slots[idx]
        slot.visits += 1
        self._maybe_schedule_replacement(slot)
        return slot.ctx, float(delta_wait)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


class _CustomFusedIterableDataset(IterableDataset):
    def __init__(self, *, cell_ctx: CellContext) -> None:
        super().__init__()
        self.cell_ctx = cell_ctx

    def _worker_shard(self, records: list[Any], worker_id: int, num_workers: int) -> list[Any]:
        if num_workers <= 1:
            return records
        shard = [r for r in records if hash(r.scan_id) % num_workers == worker_id]
        return shard or records

    def __iter__(self):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required for custom_fused backend") from exc

        worker = torch.utils.data.get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        records = self._worker_shard(self.cell_ctx.records, worker_id, num_workers)
        seed = int(self.cell_ctx.runtime.seed + worker_id * 13_337)
        rng = np.random.default_rng(seed)

        runtime = self.cell_ctx.runtime
        cell = self.cell_ctx.cell

        if cell.cache_state == "warm_pool":
            pool = _WarmPool(
                records=records,
                pool_size=runtime.pool_size,
                visits_per_scan=runtime.visits_per_scan,
                prefetch_replacements=runtime.prefetch_replacements,
                seed=seed,
            )
            pool.bootstrap()
            try:
                while True:
                    ctx, replacement_wait = pool.sample()
                    yield build_asymmetric_sample(
                        ctx=ctx,
                        n_patches=cell.n_patches,
                        device=self.cell_ctx.runtime.device,
                        rng=rng,
                        backend="custom_fused",
                        cache_state=cell.cache_state,
                        worker_id=worker_id,
                        replacement_wait_ms_delta=replacement_wait,
                    )
            finally:
                pool.close()
        else:
            while True:
                rec = records[int(rng.integers(0, len(records)))]
                try:
                    ctx = load_nifti_context(scan_id=rec.scan_id, nifti_path=rec.nifti_path)
                except Exception:
                    continue
                yield build_asymmetric_sample(
                    ctx=ctx,
                    n_patches=cell.n_patches,
                    device=self.cell_ctx.runtime.device,
                    rng=rng,
                    backend="custom_fused",
                    cache_state=cell.cache_state,
                    worker_id=worker_id,
                    replacement_wait_ms_delta=0.0,
                )


class CustomFusedBackend(BackendAdapter):
    def name(self) -> str:
        return "custom_fused"

    def prepare(self, cell_ctx: CellContext) -> None:
        del cell_ctx

    def build_dataset(self, cell_ctx: CellContext):
        return _CustomFusedIterableDataset(cell_ctx=cell_ctx)

    def teardown(self) -> None:
        return None
