"""Python wrapper around optional PyO3 medrs patch bridge."""

from __future__ import annotations

import importlib

import numpy as np

_bridge = None
_bridge_error: str | None = None


try:
    _bridge = importlib.import_module("medrs_patch_bridge")
except Exception as exc:  # pragma: no cover
    _bridge = None
    _bridge_error = str(exc)


def bridge_available() -> bool:
    return _bridge is not None


def bridge_error() -> str | None:
    return _bridge_error


def _trilinear_sample_point(volume_xyz: np.ndarray, x: float, y: float, z: float) -> float:
    sx, sy, sz = volume_xyz.shape
    if x < 0.0 or y < 0.0 or z < 0.0 or x > sx - 1 or y > sy - 1 or z > sz - 1:
        return 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    z0 = int(np.floor(z))
    x1 = min(x0 + 1, sx - 1)
    y1 = min(y0 + 1, sy - 1)
    z1 = min(z0 + 1, sz - 1)

    dx = float(x - x0)
    dy = float(y - y0)
    dz = float(z - z0)

    c000 = float(volume_xyz[x0, y0, z0])
    c100 = float(volume_xyz[x1, y0, z0])
    c010 = float(volume_xyz[x0, y1, z0])
    c110 = float(volume_xyz[x1, y1, z0])
    c001 = float(volume_xyz[x0, y0, z1])
    c101 = float(volume_xyz[x1, y0, z1])
    c011 = float(volume_xyz[x0, y1, z1])
    c111 = float(volume_xyz[x1, y1, z1])

    c00 = c000 * (1.0 - dx) + c100 * dx
    c01 = c001 * (1.0 - dx) + c101 * dx
    c10 = c010 * (1.0 - dx) + c110 * dx
    c11 = c011 * (1.0 - dx) + c111 * dx
    c0 = c00 * (1.0 - dy) + c10 * dy
    c1 = c01 * (1.0 - dy) + c11 * dy
    return c0 * (1.0 - dz) + c1 * dz


def _python_fallback_sample_patches(volume_xyz: np.ndarray, coords_xyz: np.ndarray) -> np.ndarray:
    """Fallback sampler.

    Args:
        volume_xyz: shape (X, Y, Z)
        coords_xyz: shape (P, H, W, 3)
    """
    arr = np.asarray(coords_xyz, dtype=np.float32)
    p, h, w, _ = arr.shape
    out = np.zeros((p, h, w), dtype=np.float32)
    for i in range(p):
        for yy in range(h):
            for xx in range(w):
                x, y, z = arr[i, yy, xx]
                out[i, yy, xx] = _trilinear_sample_point(volume_xyz, float(x), float(y), float(z))
    return out


def sample_patches_trilinear(volume_xyz: np.ndarray, coords_xyz: np.ndarray) -> np.ndarray:
    """Sample patches with bridge if available; otherwise use Python fallback.

    Args:
        volume_xyz: shape (X, Y, Z)
        coords_xyz: shape (P, H, W, 3)
    """
    if _bridge is not None and hasattr(_bridge, "sample_patches_trilinear"):
        return np.asarray(
            _bridge.sample_patches_trilinear(
                np.asarray(volume_xyz, dtype=np.float32),
                np.asarray(coords_xyz, dtype=np.float32),
            ),
            dtype=np.float32,
        )
    return _python_fallback_sample_patches(np.asarray(volume_xyz, dtype=np.float32), np.asarray(coords_xyz, dtype=np.float32))
