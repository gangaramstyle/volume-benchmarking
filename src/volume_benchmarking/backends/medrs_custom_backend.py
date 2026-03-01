"""MedRS+custom backend with strict medrs I/O and warm-slot replacement."""

from __future__ import annotations

import queue
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import traceback
from typing import Any

import numpy as np

from volume_benchmarking.backends.base import BackendAdapter, CellContext
from volume_benchmarking.backends.medrs_backend import (
    _DecodeResult,
    _MedRSDecoder,
    _build_asymmetric_sample_medrs,
)
from volume_benchmarking.rust_bridge import bridge_available, bridge_error

try:
    from torch.utils.data import IterableDataset
except Exception:  # pragma: no cover
    class IterableDataset:  # type: ignore
        pass


@dataclass
class _PoolSlot:
    record_idx: int
    decoded: _DecodeResult
    visits: int = 0
    replacing: bool = False
    future: Future | None = None
    future_started: float = 0.0


class _MedRSWarmPool:
    def __init__(
        self,
        *,
        records,
        decoder: _MedRSDecoder,
        pool_size: int,
        visits_per_scan: int,
        prefetch_replacements: int,
        seed: int,
    ) -> None:
        self._records = list(records)
        self._decoder = decoder
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

    def _decode_record(self, record) -> _DecodeResult:
        return self._decoder.decode(scan_id=record.scan_id, nifti_path=record.nifti_path)

    def bootstrap(self) -> None:
        attempts = 0
        max_attempts = max(len(self._records) * 3, self._pool_size)
        while len(self._slots) < self._pool_size and attempts < max_attempts:
            attempts += 1
            rec = self._next_record()
            try:
                decoded = self._decode_record(rec)
            except Exception:
                continue
            self._slots.append(_PoolSlot(record_idx=(self._next_record_cursor - 1), decoded=decoded))
        if not self._slots:
            raise RuntimeError(
                "MedRS custom warm pool bootstrap failed: no valid volume could be decoded "
                "with strict medrs I/O."
            )

    def poll_replacements(self) -> float:
        delta = 0.0
        with self._lock:
            for slot in self._slots:
                if slot.future is None or not slot.future.done():
                    continue
                wait_ms = max((time.perf_counter() - slot.future_started) * 1000.0, 0.0)
                delta += wait_ms
                try:
                    slot.decoded = slot.future.result()
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
        slot.future = self._executor.submit(self._decode_record, rec)

    def sample(self) -> tuple[_DecodeResult, float]:
        if not self._slots:
            raise RuntimeError("MedRS custom warm pool has no slots")
        delta_wait = self.poll_replacements()

        idx = self._rr_index % len(self._slots)
        self._rr_index += 1
        slot = self._slots[idx]
        slot.visits += 1
        self._maybe_schedule_replacement(slot)
        return slot.decoded, float(delta_wait)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


class _MedRSCustomIterableDataset(IterableDataset):
    def __init__(self, *, cell_ctx: CellContext) -> None:
        super().__init__()
        self.cell_ctx = cell_ctx

    def _worker_shard(self, records: list[Any], worker_id: int, num_workers: int) -> list[Any]:
        if num_workers <= 1:
            return records
        shard = [r for r in records if hash(r.scan_id) % num_workers == worker_id]
        return shard or records

    def _pool_size_for_internal_worker(self, internal_workers: int) -> int:
        total = max(1, int(self.cell_ctx.runtime.pool_size))
        return max(1, total // max(1, int(internal_workers)))

    def _sample_from_cold_records(
        self,
        *,
        records: list[Any],
        rng: np.random.Generator,
        worker_id: int,
        decoder: _MedRSDecoder,
        failed_scan_ids: set[str],
    ) -> dict[str, Any]:
        eligible = [r for r in records if r.scan_id not in failed_scan_ids]
        if not eligible:
            raise RuntimeError(
                "MedRS custom backend could not decode any eligible records in this shard. "
                "All records failed strict medrs decode at least once."
            )
        rec = eligible[int(rng.integers(0, len(eligible)))]
        try:
            decoded = decoder.decode(scan_id=rec.scan_id, nifti_path=rec.nifti_path)
        except Exception:
            failed_scan_ids.add(rec.scan_id)
            return self._sample_from_cold_records(
                records=records,
                rng=rng,
                worker_id=worker_id,
                decoder=decoder,
                failed_scan_ids=failed_scan_ids,
            )
        return _build_asymmetric_sample_medrs(
            ctx=decoded.ctx,
            n_patches=self.cell_ctx.cell.n_patches,
            rng=rng,
            cache_state=self.cell_ctx.cell.cache_state,
            worker_id=worker_id,
            io_mode=decoded.io_mode,
            backend_name="medrs_custom",
            replacement_wait_ms_delta=0.0,
        )

    def _single_thread_iter(self, worker_id: int):
        records = self._worker_shard(self.cell_ctx.records, worker_id, 1)
        seed = int(self.cell_ctx.runtime.seed + worker_id * 13_337)
        rng = np.random.default_rng(seed)
        decoder = _MedRSDecoder()
        runtime = self.cell_ctx.runtime
        cell = self.cell_ctx.cell

        if cell.cache_state == "warm_pool":
            pool = _MedRSWarmPool(
                records=records,
                decoder=decoder,
                pool_size=self._pool_size_for_internal_worker(1),
                visits_per_scan=runtime.visits_per_scan,
                prefetch_replacements=runtime.prefetch_replacements,
                seed=seed,
            )
            pool.bootstrap()
            try:
                while True:
                    decoded, replacement_wait = pool.sample()
                    yield _build_asymmetric_sample_medrs(
                        ctx=decoded.ctx,
                        n_patches=cell.n_patches,
                        rng=rng,
                        cache_state=cell.cache_state,
                        worker_id=worker_id,
                        io_mode=decoded.io_mode,
                        backend_name="medrs_custom",
                        replacement_wait_ms_delta=replacement_wait,
                    )
            finally:
                pool.close()
            return

        failed_scan_ids: set[str] = set()
        while True:
            yield self._sample_from_cold_records(
                records=records,
                rng=rng,
                worker_id=worker_id,
                decoder=decoder,
                failed_scan_ids=failed_scan_ids,
            )

    def _threaded_iter(self, internal_workers: int):
        q: queue.Queue[tuple[str, int, Any]] = queue.Queue(maxsize=max(4, internal_workers * 2))
        stop_event = threading.Event()
        timeout_s = float(max(int(self.cell_ctx.runtime.data_loader_timeout_s), 30))
        records = list(self.cell_ctx.records)
        runtime = self.cell_ctx.runtime
        cell = self.cell_ctx.cell

        def _producer(thread_idx: int) -> None:
            thread_records = self._worker_shard(records, thread_idx, internal_workers)
            seed = int(self.cell_ctx.runtime.seed + thread_idx * 13_337)
            rng = np.random.default_rng(seed)
            decoder = _MedRSDecoder()

            if cell.cache_state == "warm_pool":
                pool = _MedRSWarmPool(
                    records=thread_records,
                    decoder=decoder,
                    pool_size=self._pool_size_for_internal_worker(internal_workers),
                    visits_per_scan=runtime.visits_per_scan,
                    prefetch_replacements=runtime.prefetch_replacements,
                    seed=seed,
                )
                try:
                    pool.bootstrap()
                    while not stop_event.is_set():
                        decoded, replacement_wait = pool.sample()
                        sample = _build_asymmetric_sample_medrs(
                            ctx=decoded.ctx,
                            n_patches=cell.n_patches,
                            rng=rng,
                            cache_state=cell.cache_state,
                            worker_id=thread_idx,
                            io_mode=decoded.io_mode,
                            backend_name="medrs_custom",
                            replacement_wait_ms_delta=replacement_wait,
                        )
                        while not stop_event.is_set():
                            try:
                                q.put(("sample", thread_idx, sample), timeout=0.1)
                                break
                            except queue.Full:
                                continue
                except Exception:
                    err = traceback.format_exc()
                    stop_event.set()
                    try:
                        q.put(("error", thread_idx, err), timeout=0.1)
                    except Exception:
                        pass
                finally:
                    pool.close()
                return

            failed_scan_ids: set[str] = set()
            while not stop_event.is_set():
                try:
                    sample = self._sample_from_cold_records(
                        records=thread_records,
                        rng=rng,
                        worker_id=thread_idx,
                        decoder=decoder,
                        failed_scan_ids=failed_scan_ids,
                    )
                except Exception:
                    err = traceback.format_exc()
                    stop_event.set()
                    try:
                        q.put(("error", thread_idx, err), timeout=0.1)
                    except Exception:
                        pass
                    return

                while not stop_event.is_set():
                    try:
                        q.put(("sample", thread_idx, sample), timeout=0.1)
                        break
                    except queue.Full:
                        continue

        threads = [
            threading.Thread(target=_producer, args=(idx,), daemon=True, name=f"medrs-custom-worker-{idx}")
            for idx in range(internal_workers)
        ]
        for t in threads:
            t.start()

        try:
            while True:
                try:
                    kind, thread_idx, payload = q.get(timeout=timeout_s)
                except queue.Empty as exc:
                    if not any(t.is_alive() for t in threads):
                        raise RuntimeError("medrs_custom internal workers exited before producing samples.") from exc
                    raise RuntimeError(
                        "Timed out waiting for medrs_custom internal worker samples. "
                        "Increase runtime.data_loader_timeout_s if this is expected."
                    ) from exc
                if kind == "error":
                    raise RuntimeError(
                        f"medrs_custom internal worker {thread_idx} failed while preparing a sample:\n{payload}"
                    )
                yield payload
        finally:
            stop_event.set()
            for t in threads:
                t.join(timeout=1.0)

    def __iter__(self):
        internal_workers = int(self.cell_ctx.cell.workers)
        if internal_workers <= 0:
            yield from self._single_thread_iter(worker_id=0)
            return
        yield from self._threaded_iter(internal_workers=internal_workers)


class MedRSCustomBackend(BackendAdapter):
    def name(self) -> str:
        return "medrs_custom"

    def prepare(self, cell_ctx: CellContext) -> None:
        del cell_ctx
        if not bridge_available():
            err = bridge_error() or "unknown import error"
            raise RuntimeError(
                "MedRS custom backend hard-fail: Rust bridge is required but unavailable "
                f"({err})."
            )

    def build_dataset(self, cell_ctx: CellContext):
        return _MedRSCustomIterableDataset(cell_ctx=cell_ctx)

    def teardown(self) -> None:
        return None
