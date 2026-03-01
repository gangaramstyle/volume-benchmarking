"""MedRS backend using strict medrs I/O plus fused Rust A/B patch extraction."""

from __future__ import annotations

import collections
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from volume_benchmarking.backends.base import BackendAdapter, CellContext
from volume_benchmarking.geometry import (
    VolumeGeometry,
    clamp_world_to_volume,
    euler_xyz_to_matrix,
    sample_anchor_a_voxel,
    sample_anchor_b_world,
    sample_points_in_sphere,
    spacing_from_affine,
    voxel_to_world,
)
from volume_benchmarking.payload import (
    VolumeContext,
    robust_window_stats,
    sample_window_params,
    resolve_nifti_path,
)
from volume_benchmarking.rust_bridge import bridge_available, bridge_error, sample_asymmetric_patches

try:
    from torch.utils.data import IterableDataset
except Exception:  # pragma: no cover
    class IterableDataset:  # type: ignore
        pass


@dataclass
class _DecodeResult:
    ctx: VolumeContext
    io_mode: str


class _MedRSDecoder:
    def __init__(self) -> None:
        self._medrs = None
        self._io_mode = "medrs"
        try:
            import medrs  # type: ignore

            self._medrs = medrs
            self._io_mode = "medrs"
        except Exception as exc:
            self._medrs = None
            self._io_mode = f"unavailable:{exc}"

    def _try_medrs_decode(self, scan_id: str, nifti_path: str) -> VolumeContext | None:
        if self._medrs is None:
            return None

        try:
            resolved_path = resolve_nifti_path(nifti_path)
        except Exception:
            return None

        medrs = self._medrs
        candidates = [
            getattr(medrs, "read_nifti", None),
            getattr(medrs, "load_nifti", None),
            getattr(medrs, "load", None),
        ]
        fn = next((c for c in candidates if callable(c)), None)
        if fn is None:
            return None

        try:
            result = fn(resolved_path)
        except Exception:
            return None
        vol = None
        affine = None

        if isinstance(result, tuple) and len(result) >= 2:
            vol, affine = result[0], result[1]
        elif hasattr(result, "data"):
            vol = getattr(result, "data")
            affine = getattr(result, "affine", None)

        if vol is None:
            return None

        volume_xyz = np.asarray(vol, dtype=np.float32)
        if volume_xyz.ndim == 4:
            volume_xyz = volume_xyz[..., 0]
        if volume_xyz.ndim != 3:
            return None

        if affine is None:
            affine_arr = np.eye(4, dtype=np.float32)
        else:
            affine_arr = np.asarray(affine, dtype=np.float32)
            if affine_arr.shape != (4, 4):
                affine_arr = np.eye(4, dtype=np.float32)

        affine_inv = np.linalg.inv(affine_arr).astype(np.float32, copy=False)
        spacing = spacing_from_affine(affine_arr)
        return VolumeContext(
            scan_id=scan_id,
            volume_xyz=volume_xyz,
            affine=affine_arr,
            affine_inv=affine_inv,
            spacing_mm=spacing,
            geometry=VolumeGeometry(
                shape_xyz=tuple(int(v) for v in volume_xyz.shape),
                spacing_mm=tuple(float(v) for v in spacing.tolist()),
                affine=affine_arr,
                affine_inv=affine_inv,
            ),
        )

    def decode(self, scan_id: str, nifti_path: str) -> _DecodeResult:
        ctx = self._try_medrs_decode(scan_id, nifti_path)
        if ctx is None:
            raise RuntimeError(
                "MedRS decode failed for this record. "
                "This backend is configured for strict medrs I/O without nibabel fallback."
            )
        return _DecodeResult(ctx=ctx, io_mode="medrs")


class _MedRSLRUCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max(1, int(max_size))
        self._cache: collections.OrderedDict[str, _DecodeResult] = collections.OrderedDict()

    def get(self, key: str) -> _DecodeResult | None:
        if key not in self._cache:
            return None
        val = self._cache.pop(key)
        self._cache[key] = val
        return val

    def put(self, key: str, value: _DecodeResult) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)


def _build_asymmetric_sample_medrs(
    *,
    ctx: VolumeContext,
    n_patches: int,
    rng: np.random.Generator,
    cache_state: str,
    worker_id: int,
    io_mode: str,
) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for medrs backend") from exc

    median, std = robust_window_stats(ctx.volume_xyz)
    window_a = sample_window_params(rng, median, std)
    window_b = sample_window_params(rng, median, std)

    anchor_a_vox = sample_anchor_a_voxel(rng, ctx.geometry.shape_xyz, ctx.spacing_mm)
    anchor_a_world = voxel_to_world(anchor_a_vox[np.newaxis, :], ctx.affine)[0]
    anchor_b_world = sample_anchor_b_world(rng, anchor_a_world, ctx.geometry)

    rotation_euler_deg = rng.uniform(-20.0, 20.0, size=3).astype(np.float32)
    rotation_b = euler_xyz_to_matrix(rotation_euler_deg)

    radius_a = float(rng.uniform(20.0, 30.0))
    radius_b = float(rng.uniform(20.0, 30.0))
    rel_coords_a = sample_points_in_sphere(rng, n_patches, radius_a)
    rel_coords_b = sample_points_in_sphere(rng, n_patches, radius_b)

    centers_a_world = anchor_a_world[np.newaxis, :] + rel_coords_a
    centers_b_world = anchor_b_world[np.newaxis, :] + rel_coords_b
    for idx in range(centers_a_world.shape[0]):
        centers_a_world[idx] = clamp_world_to_volume(centers_a_world[idx], ctx.geometry)
    for idx in range(centers_b_world.shape[0]):
        centers_b_world[idx] = clamp_world_to_volume(centers_b_world[idx], ctx.geometry)

    patches_a, patches_b = sample_asymmetric_patches(
        volume_xyz=ctx.volume_xyz,
        affine_inv=ctx.affine_inv,
        centers_a_world=centers_a_world,
        centers_b_world=centers_b_world,
        rotation_matrix=rotation_b,
        window_a_wc=window_a.wc,
        window_a_ww=window_a.ww,
        window_b_wc=window_b.wc,
        window_b_ww=window_b.ww,
        a_native_no_interp=True,
    )

    return {
        "patches_a": torch.from_numpy(patches_a).unsqueeze(1).float(),
        "patches_b": torch.from_numpy(patches_b).unsqueeze(1).float(),
        "rel_coords_a_mm": torch.from_numpy(rel_coords_a.astype(np.float32, copy=False)),
        "rel_coords_b_mm": torch.from_numpy(rel_coords_b.astype(np.float32, copy=False)),
        "anchor_delta_mm_a_frame": torch.from_numpy((anchor_b_world - anchor_a_world).astype(np.float32, copy=False)),
        "rotation_b_euler_deg": torch.from_numpy(rotation_euler_deg.astype(np.float32, copy=False)),
        "rotation_b_matrix": torch.from_numpy(rotation_b.astype(np.float32, copy=False)),
        "meta": {
            "scan_id": ctx.scan_id,
            "backend": "medrs",
            "cache_state": cache_state,
            "worker_id": int(worker_id),
            "window_a": {"wc": window_a.wc, "ww": window_a.ww},
            "window_b": {"wc": window_b.wc, "ww": window_b.ww},
            "replacement_wait_ms_delta": 0.0,
            "medrs_io_mode": io_mode,
            "medrs_bridge_available": bool(bridge_available()),
            "medrs_bridge_error": bridge_error(),
            "medrs_path": "rust_fused_ab_sampler",
            "medrs_a_sampler_mode": "native_no_interp",
            "medrs_windowing_mode": "rust_fused",
        },
    }


class _MedRSIterableDataset(IterableDataset):
    def __init__(self, *, cell_ctx: CellContext) -> None:
        super().__init__()
        self.cell_ctx = cell_ctx

    def _worker_shard(self, records, worker_id: int, num_workers: int):
        if num_workers <= 1:
            return records
        shard = [r for r in records if hash(r.scan_id) % num_workers == worker_id]
        return shard or records

    def _build_sample(
        self,
        *,
        records,
        rng: np.random.Generator,
        worker_id: int,
        decoder: _MedRSDecoder,
        cache: _MedRSLRUCache | None,
        failed_scan_ids: set[str],
    ) -> dict[str, Any]:
        eligible = [r for r in records if r.scan_id not in failed_scan_ids]
        if not eligible:
            raise RuntimeError(
                "MedRS backend could not decode any eligible records in this shard. "
                "All records failed at least once; consider a cleaner catalog subset."
            )

        perm = rng.permutation(len(eligible))
        decoded: _DecodeResult | None = None
        for idx in perm:
            rec = eligible[int(idx)]
            cached = cache.get(rec.scan_id) if cache is not None else None
            if cached is None:
                try:
                    decoded = decoder.decode(scan_id=rec.scan_id, nifti_path=rec.nifti_path)
                except Exception:
                    failed_scan_ids.add(rec.scan_id)
                    continue
                if cache is not None:
                    cache.put(rec.scan_id, decoded)
            else:
                decoded = cached
            break

        if decoded is None:
            raise RuntimeError(
                "MedRS backend could not decode a record for this sample. "
                "All currently eligible records failed."
            )

        return _build_asymmetric_sample_medrs(
            ctx=decoded.ctx,
            n_patches=self.cell_ctx.cell.n_patches,
            rng=rng,
            cache_state=self.cell_ctx.cell.cache_state,
            worker_id=worker_id,
            io_mode=decoded.io_mode,
        )

    def _single_thread_iter(self, worker_id: int):
        records = self._worker_shard(self.cell_ctx.records, worker_id, 1)
        rng = np.random.default_rng(self.cell_ctx.runtime.seed + 500 + worker_id)
        decoder = _MedRSDecoder()
        use_cache = self.cell_ctx.cell.cache_state == "warm_pool"
        cache = _MedRSLRUCache(max_size=self.cell_ctx.runtime.pool_size) if use_cache else None
        failed_scan_ids: set[str] = set()

        while True:
            yield self._build_sample(
                records=records,
                rng=rng,
                worker_id=worker_id,
                decoder=decoder,
                cache=cache,
                failed_scan_ids=failed_scan_ids,
            )

    def _threaded_iter(self, internal_workers: int):
        q: queue.Queue[tuple[str, int, Any]] = queue.Queue(maxsize=max(4, internal_workers * 2))
        stop_event = threading.Event()
        timeout_s = float(max(int(self.cell_ctx.runtime.data_loader_timeout_s), 30))
        records = list(self.cell_ctx.records)

        def _producer(thread_idx: int) -> None:
            rng = np.random.default_rng(self.cell_ctx.runtime.seed + 500 + thread_idx)
            decoder = _MedRSDecoder()
            use_cache = self.cell_ctx.cell.cache_state == "warm_pool"
            cache = _MedRSLRUCache(max_size=self.cell_ctx.runtime.pool_size) if use_cache else None
            failed_scan_ids: set[str] = set()
            thread_records = self._worker_shard(records, thread_idx, internal_workers)

            while not stop_event.is_set():
                try:
                    sample = self._build_sample(
                        records=thread_records,
                        rng=rng,
                        worker_id=thread_idx,
                        decoder=decoder,
                        cache=cache,
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
            threading.Thread(target=_producer, args=(idx,), daemon=True, name=f"medrs-worker-{idx}")
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
                        raise RuntimeError("MedRS internal workers exited before producing samples.") from exc
                    raise RuntimeError(
                        "Timed out waiting for MedRS internal worker samples. "
                        "Increase runtime.data_loader_timeout_s if this is expected."
                    ) from exc

                if kind == "error":
                    raise RuntimeError(
                        f"MedRS internal worker {thread_idx} failed while preparing a sample:\n{payload}"
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


class MedRSBackend(BackendAdapter):
    def name(self) -> str:
        return "medrs"

    def prepare(self, cell_ctx: CellContext) -> None:
        del cell_ctx
        if not bridge_available():
            err = bridge_error() or "unknown import error"
            raise RuntimeError(
                "MedRS backend hard-fail: Rust bridge is required but unavailable "
                f"({err})."
            )

    def build_dataset(self, cell_ctx: CellContext):
        return _MedRSIterableDataset(cell_ctx=cell_ctx)

    def teardown(self) -> None:
        return None
