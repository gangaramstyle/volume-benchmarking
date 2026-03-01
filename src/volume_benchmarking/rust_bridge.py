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
    return (
        _bridge is not None
        and hasattr(_bridge, "sample_patches_trilinear")
        and hasattr(_bridge, "sample_asymmetric_patches")
    )


def bridge_error() -> str | None:
    if _bridge is not None:
        missing: list[str] = []
        if not hasattr(_bridge, "sample_patches_trilinear"):
            missing.append("sample_patches_trilinear")
        if not hasattr(_bridge, "sample_asymmetric_patches"):
            missing.append("sample_asymmetric_patches")
        if missing:
            return f"medrs_patch_bridge missing symbols: {', '.join(missing)}"
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


def sample_asymmetric_patches(
    volume_xyz: np.ndarray,
    affine_inv: np.ndarray,
    centers_a_world: np.ndarray,
    centers_b_world: np.ndarray,
    rotation_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample paired A/B 2D patches with world->voxel mapping in Rust.

    Args:
        volume_xyz: shape (X, Y, Z)
        affine_inv: shape (4, 4)
        centers_a_world: shape (N, 3)
        centers_b_world: shape (N, 3)
        rotation_matrix: shape (3, 3), applied to B patch plane offsets
    """
    if not bridge_available():
        err = bridge_error() or "unknown import error"
        raise RuntimeError(
            f"MedRS bridge hard-fail: medrs_patch_bridge is unavailable ({err})."
        )
    patches_a, patches_b = _bridge.sample_asymmetric_patches(  # type: ignore[union-attr]
        np.asarray(volume_xyz, dtype=np.float32),
        np.asarray(affine_inv, dtype=np.float32),
        np.asarray(centers_a_world, dtype=np.float32),
        np.asarray(centers_b_world, dtype=np.float32),
        np.asarray(rotation_matrix, dtype=np.float32),
    )
    return (
        np.asarray(patches_a, dtype=np.float32),
        np.asarray(patches_b, dtype=np.float32),
    )
