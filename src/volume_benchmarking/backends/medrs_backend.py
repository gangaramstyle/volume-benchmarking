"""MedRS backend with optional Rust bridge for coordinate-mapped patch extraction."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

import numpy as np

from volume_benchmarking.backends.base import BackendAdapter, CellContext
from volume_benchmarking.geometry import VolumeGeometry, spacing_from_affine, world_to_voxel
from volume_benchmarking.payload import (
    VolumeContext,
    build_asymmetric_sample,
    load_nifti_context,
    resolve_nifti_path,
)
from volume_benchmarking.rust_bridge import bridge_available, bridge_error, sample_patches_trilinear

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
        self._io_mode = "nibabel_fallback"
        try:
            import medrs  # type: ignore

            self._medrs = medrs
            self._io_mode = "medrs"
        except Exception:
            self._medrs = None
            self._io_mode = "nibabel_fallback"

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
        if ctx is not None:
            return _DecodeResult(ctx=ctx, io_mode="medrs")
        return _DecodeResult(ctx=load_nifti_context(scan_id=scan_id, nifti_path=nifti_path), io_mode="nibabel_fallback")


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


def _medrs_bridge_patch_extractor_batch(
    volume_zyx_tensor,
    affine_inv: np.ndarray,
    shape_xyz: tuple[int, int, int],
    centers_world_mm: np.ndarray,
    rotation_matrix: np.ndarray,
    plane_offsets_mm: np.ndarray,
) -> np.ndarray:
    del shape_xyz
    vol_xyz = volume_zyx_tensor[0, 0].detach().cpu().numpy().transpose(2, 1, 0).astype(np.float32, copy=False)
    centers = np.asarray(centers_world_mm, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers_world_mm must be shape (N, 3), got {centers.shape}")

    offsets = plane_offsets_mm.reshape(-1, 3)
    rotated = (offsets @ rotation_matrix.T).astype(np.float32, copy=False)
    points_world = centers[:, np.newaxis, :] + rotated[np.newaxis, :, :]
    points_vox = world_to_voxel(points_world.reshape(-1, 3), affine_inv).reshape(centers.shape[0], 16, 16, 3)
    sampled = sample_patches_trilinear(vol_xyz, points_vox)
    return sampled.astype(np.float32, copy=False)


class _MedRSIterableDataset(IterableDataset):
    def __init__(self, *, cell_ctx: CellContext) -> None:
        super().__init__()
        self.cell_ctx = cell_ctx

    def _worker_shard(self, records, worker_id: int, num_workers: int):
        if num_workers <= 1:
            return records
        shard = [r for r in records if hash(r.scan_id) % num_workers == worker_id]
        return shard or records

    def __iter__(self):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required for medrs backend") from exc

        worker = torch.utils.data.get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        rng = np.random.default_rng(self.cell_ctx.runtime.seed + 500 + worker_id)
        records = self._worker_shard(self.cell_ctx.records, worker_id, num_workers)
        failed_scan_ids: set[str] = set()

        decoder = _MedRSDecoder()
        use_cache = self.cell_ctx.cell.cache_state == "warm_pool"
        cache = _MedRSLRUCache(max_size=self.cell_ctx.runtime.pool_size) if use_cache else None

        while True:
            eligible = [r for r in records if r.scan_id not in failed_scan_ids]
            if not eligible:
                raise RuntimeError(
                    "MedRS backend could not decode any eligible records in this shard. "
                    "All records failed at least once; consider a cleaner catalog subset."
                )

            rec = eligible[int(rng.integers(0, len(eligible)))]
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

            sample = build_asymmetric_sample(
                ctx=decoded.ctx,
                n_patches=self.cell_ctx.cell.n_patches,
                device=self.cell_ctx.runtime.device,
                rng=rng,
                backend="medrs",
                cache_state=self.cell_ctx.cell.cache_state,
                worker_id=worker_id,
                replacement_wait_ms_delta=0.0,
                b_extractor_batch=_medrs_bridge_patch_extractor_batch,
            )
            sample["meta"]["medrs_io_mode"] = decoded.io_mode
            sample["meta"]["medrs_bridge_available"] = bool(bridge_available())
            sample["meta"]["medrs_bridge_error"] = bridge_error()
            yield sample


class MedRSBackend(BackendAdapter):
    def name(self) -> str:
        return "medrs"

    def prepare(self, cell_ctx: CellContext) -> None:
        del cell_ctx

    def build_dataset(self, cell_ctx: CellContext):
        return _MedRSIterableDataset(cell_ctx=cell_ctx)

    def teardown(self) -> None:
        return None
