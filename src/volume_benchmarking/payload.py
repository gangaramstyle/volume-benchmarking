"""Asymmetric payload generation and patch extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from volume_benchmarking.geometry import (
    VolumeGeometry,
    clamp_world_to_volume,
    euler_xyz_to_matrix,
    patch_plane_offsets_mm,
    sample_anchor_a_voxel,
    sample_anchor_b_world,
    sample_points_in_sphere,
    spacing_from_affine,
    voxel_to_world,
    world_to_voxel,
)


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("torch is required for payload extraction") from exc
    return torch, F


def _require_nibabel():
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("nibabel is required to load NIfTI volumes") from exc
    return nib


def resolve_nifti_path(path_or_series: str) -> str:
    """Resolve a concrete NIfTI file path from a file or series directory."""
    p = Path(path_or_series).expanduser()
    if p.is_file():
        return str(p)

    if p.is_dir():
        candidates = sorted(
            [*p.glob("*.nii.gz"), *p.glob("*.nii")],
            key=lambda f: f.stat().st_size if f.exists() else 0,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    raise FileNotFoundError(f"Could not resolve NIfTI file from path: {path_or_series}")


@dataclass
class VolumeContext:
    scan_id: str
    volume_xyz: np.ndarray
    affine: np.ndarray
    affine_inv: np.ndarray
    spacing_mm: np.ndarray
    geometry: VolumeGeometry


@dataclass(frozen=True)
class WindowParams:
    wc: float
    ww: float


def load_nifti_context(scan_id: str, nifti_path: str) -> VolumeContext:
    nib = _require_nibabel()
    resolved = resolve_nifti_path(nifti_path)
    img = nib.load(resolved)
    try:
        img = nib.as_closest_canonical(img)
    except Exception:
        pass
    volume = np.asarray(img.get_fdata(), dtype=np.float32)
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape={volume.shape} for {nifti_path}")

    affine = np.asarray(img.affine, dtype=np.float32)
    affine_inv = np.linalg.inv(affine).astype(np.float32, copy=False)
    spacing = spacing_from_affine(affine)
    geom = VolumeGeometry(
        shape_xyz=tuple(int(v) for v in volume.shape),
        spacing_mm=tuple(float(v) for v in spacing.tolist()),
        affine=affine,
        affine_inv=affine_inv,
    )
    return VolumeContext(
        scan_id=scan_id,
        volume_xyz=volume,
        affine=affine,
        affine_inv=affine_inv,
        spacing_mm=spacing,
        geometry=geom,
    )


def robust_window_stats(volume_xyz: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(volume_xyz, dtype=np.float32)
    p_low, p_high = np.percentile(arr, [0.5, 99.5])
    clipped = np.clip(arr, p_low, p_high)
    median = float(np.median(clipped))
    std = float(np.std(clipped))
    if std <= 1e-6:
        std = 1.0
    return median, std


def sample_window_params(rng: np.random.Generator, median: float, std: float) -> WindowParams:
    wc = float(rng.uniform(median - std, median + std))
    ww = float(rng.uniform(2.0 * std, 6.0 * std))
    ww = max(ww, 1e-3)
    return WindowParams(wc=wc, ww=ww)


def apply_window(patches: np.ndarray, wc: float, ww: float) -> np.ndarray:
    wmin = wc - 0.5 * ww
    wmax = wc + 0.5 * ww
    clipped = np.clip(patches, wmin, wmax)
    norm = ((clipped - wmin) / max(wmax - wmin, 1e-6)) * 2.0 - 1.0
    return norm.astype(np.float32, copy=False)


def _resize_2d_patch(patch_2d: np.ndarray, out_size: int = 16) -> np.ndarray:
    torch, F = _require_torch()
    t = torch.from_numpy(np.asarray(patch_2d, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def extract_native_patch_a(
    volume_xyz: np.ndarray,
    affine_inv: np.ndarray,
    spacing_mm: np.ndarray,
    center_world_mm: np.ndarray,
    source_mm: tuple[float, float, float] = (32.0, 32.0, 1.0),
) -> np.ndarray:
    center_vox = world_to_voxel(center_world_mm[np.newaxis, :], affine_inv)[0]
    center_idx = np.rint(center_vox).astype(np.int64)

    block_vox = np.maximum(np.rint(np.asarray(source_mm, dtype=np.float32) / np.clip(spacing_mm, 1e-6, None)).astype(np.int64), 1)
    starts = center_idx - (block_vox // 2)
    ends = starts + block_vox

    shape = np.asarray(volume_xyz.shape, dtype=np.int64)
    src_starts = np.maximum(starts, 0)
    src_ends = np.minimum(ends, shape)

    patch = np.zeros(tuple(int(v) for v in block_vox.tolist()), dtype=np.float32)
    dst_starts = src_starts - starts
    dst_ends = dst_starts + (src_ends - src_starts)

    patch[
        int(dst_starts[0]) : int(dst_ends[0]),
        int(dst_starts[1]) : int(dst_ends[1]),
        int(dst_starts[2]) : int(dst_ends[2]),
    ] = volume_xyz[
        int(src_starts[0]) : int(src_ends[0]),
        int(src_starts[1]) : int(src_ends[1]),
        int(src_starts[2]) : int(src_ends[2]),
    ]

    z_idx = int(patch.shape[2] // 2)
    patch_2d = patch[:, :, z_idx]
    return _resize_2d_patch(patch_2d, out_size=16)


def _volume_xyz_to_zyx_tensor(volume_xyz: np.ndarray, device: str):
    torch, _ = _require_torch()
    vol_zyx = np.transpose(np.asarray(volume_xyz, dtype=np.float32), (2, 1, 0))
    t = torch.from_numpy(vol_zyx).unsqueeze(0).unsqueeze(0).to(device)
    return t


def _coords_world_to_grid(
    points_world_mm: np.ndarray,
    affine_inv: np.ndarray,
    shape_xyz: tuple[int, int, int],
):
    torch, _ = _require_torch()
    vox_xyz = world_to_voxel(points_world_mm, affine_inv)
    sx, sy, sz = [float(v) for v in shape_xyz]

    x = vox_xyz[:, 0]
    y = vox_xyz[:, 1]
    z = vox_xyz[:, 2]

    x_norm = (2.0 * (x / max(sx - 1.0, 1.0))) - 1.0
    y_norm = (2.0 * (y / max(sy - 1.0, 1.0))) - 1.0
    z_norm = (2.0 * (z / max(sz - 1.0, 1.0))) - 1.0

    grid = np.stack([x_norm, y_norm, z_norm], axis=-1).astype(np.float32, copy=False)
    return torch.from_numpy(grid)


def extract_rotated_patch_b_grid_sample(
    volume_zyx_tensor,
    affine_inv: np.ndarray,
    shape_xyz: tuple[int, int, int],
    center_world_mm: np.ndarray,
    rotation_matrix: np.ndarray,
    plane_offsets_mm: np.ndarray,
) -> np.ndarray:
    torch, F = _require_torch()
    offsets = plane_offsets_mm.reshape(-1, 3)
    rotated = (offsets @ rotation_matrix.T).astype(np.float32, copy=False)
    points_world = center_world_mm[np.newaxis, :] + rotated

    grid_flat = _coords_world_to_grid(points_world, affine_inv, shape_xyz).to(volume_zyx_tensor.device)
    grid = grid_flat.view(1, 1, 16, 16, 3)

    sampled = F.grid_sample(
        volume_zyx_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    patch = sampled[0, 0, 0]
    return patch.detach().cpu().numpy().astype(np.float32, copy=False)


def _monai_resample_patch(
    volume_zyx_tensor,
    affine_inv: np.ndarray,
    shape_xyz: tuple[int, int, int],
    center_world_mm: np.ndarray,
    rotation_matrix: np.ndarray,
    plane_offsets_mm: np.ndarray,
) -> np.ndarray:
    """Strict MONAI Resample patch extraction.

    This path is intentionally strict: if MONAI resampling is unavailable or fails,
    we raise instead of silently falling back to torch grid_sample.
    """
    try:
        from monai.transforms import Resample  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "MONAI backend hard-fail: monai.transforms.Resample is unavailable."
        ) from exc

    try:
        torch, _ = _require_torch()
        offsets = plane_offsets_mm.reshape(-1, 3)
        rotated = (offsets @ rotation_matrix.T).astype(np.float32, copy=False)
        points_world = center_world_mm[np.newaxis, :] + rotated
        grid_flat = _coords_world_to_grid(points_world, affine_inv, shape_xyz).to(volume_zyx_tensor.device)

        # MONAI Resample expects (spatial_dims, ...). We construct (3, 1, 16, 16)
        grid = grid_flat.view(16, 16, 3).permute(2, 0, 1).unsqueeze(1)
        resampler = Resample(mode="bilinear", padding_mode="zeros", align_corners=True)
        out = resampler(volume_zyx_tensor[0], grid)
        out = out.squeeze(0).squeeze(0)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as exc:
        raise RuntimeError(
            "MONAI backend hard-fail: coordinate patch resampling failed; "
            "no grid_sample fallback is allowed."
        ) from exc


def build_asymmetric_sample(
    ctx: VolumeContext,
    n_patches: int,
    device: str,
    rng: np.random.Generator,
    backend: str,
    cache_state: str,
    worker_id: int,
    replacement_wait_ms_delta: float = 0.0,
    b_extractor: Callable[..., np.ndarray] | None = None,
    b_extractor_batch: Callable[..., np.ndarray] | None = None,
) -> dict[str, object]:
    """Build one benchmark sample containing A/B patch sets and supervision targets."""
    torch, _ = _require_torch()

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

    plane_offsets = patch_plane_offsets_mm(size=16, extent_x_mm=32.0, extent_y_mm=32.0)

    patches_a = np.stack(
        [
            extract_native_patch_a(
                volume_xyz=ctx.volume_xyz,
                affine_inv=ctx.affine_inv,
                spacing_mm=ctx.spacing_mm,
                center_world_mm=centers_a_world[idx],
            )
            for idx in range(centers_a_world.shape[0])
        ],
        axis=0,
    )

    volume_zyx_tensor = _volume_xyz_to_zyx_tensor(ctx.volume_xyz, device=device)
    if b_extractor_batch is not None:
        patches_b = b_extractor_batch(
            volume_zyx_tensor=volume_zyx_tensor,
            affine_inv=ctx.affine_inv,
            shape_xyz=ctx.geometry.shape_xyz,
            centers_world_mm=centers_b_world,
            rotation_matrix=rotation_b,
            plane_offsets_mm=plane_offsets,
        )
    else:
        extractor = b_extractor or extract_rotated_patch_b_grid_sample
        patches_b = np.stack(
            [
                extractor(
                    volume_zyx_tensor=volume_zyx_tensor,
                    affine_inv=ctx.affine_inv,
                    shape_xyz=ctx.geometry.shape_xyz,
                    center_world_mm=centers_b_world[idx],
                    rotation_matrix=rotation_b,
                    plane_offsets_mm=plane_offsets,
                )
                for idx in range(centers_b_world.shape[0])
            ],
            axis=0,
        )

    patches_a = apply_window(patches_a, wc=window_a.wc, ww=window_a.ww)
    patches_b = apply_window(patches_b, wc=window_b.wc, ww=window_b.ww)

    patches_a_t = torch.from_numpy(patches_a).unsqueeze(1).float()
    patches_b_t = torch.from_numpy(patches_b).unsqueeze(1).float()
    rel_a_t = torch.from_numpy(rel_coords_a.astype(np.float32, copy=False))
    rel_b_t = torch.from_numpy(rel_coords_b.astype(np.float32, copy=False))
    anchor_delta = torch.from_numpy((anchor_b_world - anchor_a_world).astype(np.float32, copy=False))
    rot_euler = torch.from_numpy(rotation_euler_deg.astype(np.float32, copy=False))
    rot_matrix = torch.from_numpy(rotation_b.astype(np.float32, copy=False))

    return {
        "patches_a": patches_a_t,
        "patches_b": patches_b_t,
        "rel_coords_a_mm": rel_a_t,
        "rel_coords_b_mm": rel_b_t,
        "anchor_delta_mm_a_frame": anchor_delta,
        "rotation_b_euler_deg": rot_euler,
        "rotation_b_matrix": rot_matrix,
        "meta": {
            "scan_id": ctx.scan_id,
            "backend": backend,
            "cache_state": cache_state,
            "worker_id": int(worker_id),
            "window_a": {"wc": window_a.wc, "ww": window_a.ww},
            "window_b": {"wc": window_b.wc, "ww": window_b.ww},
            "replacement_wait_ms_delta": float(replacement_wait_ms_delta),
        },
    }


__all__ = [
    "VolumeContext",
    "WindowParams",
    "load_nifti_context",
    "build_asymmetric_sample",
    "extract_native_patch_a",
    "extract_rotated_patch_b_grid_sample",
    "_monai_resample_patch",
]
