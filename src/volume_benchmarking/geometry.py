"""Geometry helpers for anchor/patch sampling and coordinate transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VolumeGeometry:
    shape_xyz: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    affine: np.ndarray
    affine_inv: np.ndarray


def spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    linear = np.asarray(affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(linear, axis=0)
    spacing = np.clip(spacing, 1e-6, None)
    return spacing.astype(np.float32)


def voxel_to_world(points_xyz_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz_vox, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    linear = np.asarray(affine[:3, :3], dtype=np.float32)
    trans = np.asarray(affine[:3, 3], dtype=np.float32)
    return (pts @ linear.T + trans[np.newaxis, :]).astype(np.float32, copy=False)


def world_to_voxel(points_xyz_mm: np.ndarray, affine_inv: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz_mm, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    linear = np.asarray(affine_inv[:3, :3], dtype=np.float32)
    trans = np.asarray(affine_inv[:3, 3], dtype=np.float32)
    return (pts @ linear.T + trans[np.newaxis, :]).astype(np.float32, copy=False)


def euler_xyz_to_matrix(degrees_xyz: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    x, y, z = [math.radians(float(v)) for v in degrees_xyz]
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32, copy=False)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    if n <= 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def sample_points_in_sphere(rng: np.random.Generator, n: int, radius_mm: float) -> np.ndarray:
    points = np.zeros((int(n), 3), dtype=np.float32)
    for idx in range(int(n)):
        direction = random_unit_vector(rng)
        distance = float(rng.uniform(0.0, max(radius_mm, 1e-6)))
        points[idx] = direction * distance
    return points


def sample_anchor_a_voxel(
    rng: np.random.Generator,
    shape_xyz: tuple[int, int, int],
    spacing_mm: np.ndarray,
    safety_radius_mm: float = 46.0,
) -> np.ndarray:
    shape = np.asarray(shape_xyz, dtype=np.int64)
    margin = np.ceil(safety_radius_mm / np.clip(spacing_mm, 1e-6, None)).astype(np.int64)
    low = np.maximum(margin, 0)
    high = np.maximum(shape - margin - 1, low + 1)
    out = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        lo = int(low[axis])
        hi = int(high[axis])
        if hi <= lo:
            lo = 0
            hi = int(shape[axis]) - 1
        out[axis] = float(rng.integers(lo, hi + 1))
    return out


def clamp_world_to_volume(world_point_mm: np.ndarray, geometry: VolumeGeometry) -> np.ndarray:
    vox = world_to_voxel(world_point_mm[np.newaxis, :], geometry.affine_inv)[0]
    shape = np.asarray(geometry.shape_xyz, dtype=np.float32)
    vox = np.clip(vox, 0.0, shape - 1.0)
    return voxel_to_world(vox[np.newaxis, :], geometry.affine)[0]


def sample_anchor_b_world(
    rng: np.random.Generator,
    anchor_a_world_mm: np.ndarray,
    geometry: VolumeGeometry,
    min_radius_mm: float = 20.0,
    max_radius_mm: float = 30.0,
) -> np.ndarray:
    radius = float(rng.uniform(min_radius_mm, max_radius_mm))
    direction = random_unit_vector(rng)
    raw = anchor_a_world_mm + direction * radius
    return clamp_world_to_volume(raw.astype(np.float32), geometry)


def patch_plane_offsets_mm(size: int = 16, extent_x_mm: float = 32.0, extent_y_mm: float = 32.0) -> np.ndarray:
    step_x = extent_x_mm / float(size)
    step_y = extent_y_mm / float(size)
    xs = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) * step_x
    ys = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) * step_y
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    zeros = np.zeros_like(grid_x)
    offsets = np.stack([grid_x, grid_y, zeros], axis=-1)
    return offsets.astype(np.float32, copy=False)
