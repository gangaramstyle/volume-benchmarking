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
    return _bridge is not None and hasattr(_bridge, "sample_patches_trilinear")


def bridge_error() -> str | None:
    if _bridge is not None and not hasattr(_bridge, "sample_patches_trilinear"):
        return "medrs_patch_bridge missing sample_patches_trilinear symbol"
    return _bridge_error


def sample_patches_trilinear(volume_xyz: np.ndarray, coords_xyz: np.ndarray) -> np.ndarray:
    """Sample patches through the Rust bridge.

    Args:
        volume_xyz: shape (X, Y, Z)
        coords_xyz: shape (P, H, W, 3)
    """
    if not bridge_available():
        err = bridge_error() or "unknown import error"
        raise RuntimeError(
            f"MedRS bridge hard-fail: medrs_patch_bridge is unavailable ({err})."
        )
    return np.asarray(
        _bridge.sample_patches_trilinear(  # type: ignore[union-attr]
            np.asarray(volume_xyz, dtype=np.float32),
            np.asarray(coords_xyz, dtype=np.float32),
        ),
        dtype=np.float32,
    )
