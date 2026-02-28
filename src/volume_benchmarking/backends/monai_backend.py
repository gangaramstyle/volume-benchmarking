"""MONAI backend using LoadImaged + cache datasets + coordinate patch transform."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from volume_benchmarking.backends.base import BackendAdapter, CellContext
from volume_benchmarking.geometry import VolumeGeometry, spacing_from_affine
from volume_benchmarking.payload import VolumeContext, _monai_resample_patch, build_asymmetric_sample

try:
    from torch.utils.data import IterableDataset
except Exception:  # pragma: no cover
    class IterableDataset:  # type: ignore
        pass


class _MONAICoordinatePatchTransform:
    """Custom transform that samples only requested patches instead of rotating full volume."""

    def __init__(self, *, n_patches: int, device: str, seed: int, cache_state: str) -> None:
        self.n_patches = int(n_patches)
        self.device = str(device)
        self.cache_state = str(cache_state)
        self._rng = np.random.default_rng(int(seed))
        self._lock = threading.Lock()

    def _extract_affine(self, image_obj, data: dict[str, Any]) -> np.ndarray:
        if hasattr(image_obj, "meta"):
            maybe = image_obj.meta.get("affine")
            if maybe is not None:
                return np.asarray(maybe, dtype=np.float32)
        meta = data.get("image_meta_dict", {})
        maybe = meta.get("affine") if isinstance(meta, dict) else None
        if maybe is not None:
            return np.asarray(maybe, dtype=np.float32)
        return np.eye(4, dtype=np.float32)

    def _extract_volume(self, image_obj) -> np.ndarray:
        if hasattr(image_obj, "detach"):
            arr = image_obj.detach().cpu().numpy()
        else:
            arr = np.asarray(image_obj)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D tensor from MONAI loader, got {arr.shape}")
        return np.asarray(arr, dtype=np.float32)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image_obj = data["image"]
        volume_xyz = self._extract_volume(image_obj)
        affine = self._extract_affine(image_obj, data)
        affine_inv = np.linalg.inv(affine).astype(np.float32, copy=False)
        spacing = spacing_from_affine(affine)

        scan_id = str(data.get("scan_id", "unknown_scan"))
        ctx = VolumeContext(
            scan_id=scan_id,
            volume_xyz=volume_xyz,
            affine=affine,
            affine_inv=affine_inv,
            spacing_mm=spacing,
            geometry=VolumeGeometry(
                shape_xyz=tuple(int(v) for v in volume_xyz.shape),
                spacing_mm=tuple(float(v) for v in spacing.tolist()),
                affine=affine,
                affine_inv=affine_inv,
            ),
        )

        with self._lock:
            sample = build_asymmetric_sample(
                ctx=ctx,
                n_patches=self.n_patches,
                device=self.device,
                rng=self._rng,
                backend="monai",
                cache_state=self.cache_state,
                worker_id=0,
                replacement_wait_ms_delta=0.0,
                b_extractor=_monai_resample_patch,
            )
        return sample


class _MONAIIterableDataset(IterableDataset):
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
            from monai.data import CacheDataset, Dataset
            from monai.transforms import Compose, EnsureTyped, LoadImaged
        except Exception as exc:
            raise RuntimeError("MONAI backend requires monai and torch") from exc

        worker = torch.utils.data.get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        records = self._worker_shard(self.cell_ctx.records, worker_id, num_workers)
        data = [{"image": r.nifti_path, "scan_id": r.scan_id} for r in records]

        transform = Compose(
            [
                LoadImaged(keys=["image"], image_only=False),
                EnsureTyped(keys=["image"], track_meta=True),
                _MONAICoordinatePatchTransform(
                    n_patches=self.cell_ctx.cell.n_patches,
                    device=self.cell_ctx.runtime.device,
                    seed=self.cell_ctx.runtime.seed + worker_id,
                    cache_state=self.cell_ctx.cell.cache_state,
                ),
            ]
        )

        if self.cell_ctx.cell.cache_state == "warm_pool":
            ds = CacheDataset(data=data, transform=transform, cache_rate=1.0, num_workers=0)
        else:
            ds = Dataset(data=data, transform=transform)

        rng = np.random.default_rng(self.cell_ctx.runtime.seed + worker_id + 99)
        while True:
            idx = int(rng.integers(0, len(ds)))
            sample = ds[idx]
            sample["meta"]["worker_id"] = int(worker_id)
            yield sample


class MONAIBackend(BackendAdapter):
    def name(self) -> str:
        return "monai"

    def prepare(self, cell_ctx: CellContext) -> None:
        del cell_ctx

    def build_dataset(self, cell_ctx: CellContext):
        return _MONAIIterableDataset(cell_ctx=cell_ctx)

    def teardown(self) -> None:
        return None
