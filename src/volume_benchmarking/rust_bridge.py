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
        and hasattr(_bridge, "sample_asymmetric_patches_fused")
    )


def bridge_error() -> str | None:
    if _bridge is not None:
        missing: list[str] = []
        if not hasattr(_bridge, "sample_patches_trilinear"):
            missing.append("sample_patches_trilinear")
        if not hasattr(_bridge, "sample_asymmetric_patches_fused"):
            missing.append("sample_asymmetric_patches_fused")
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
    window_a_wc: float,
    window_a_ww: float,
    window_b_wc: float,
    window_b_ww: float,
    a_native_no_interp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample paired A/B patches with fused Rust extraction + windowing.

    Args:
        volume_xyz: shape (X, Y, Z)
        affine_inv: shape (4, 4)
        centers_a_world: shape (N, 3)
        centers_b_world: shape (N, 3)
        rotation_matrix: shape (3, 3), applied to B patch plane offsets
        window_a_wc/window_a_ww: A window center/width
        window_b_wc/window_b_ww: B window center/width
        a_native_no_interp: True for nearest-neighbor A path, False for trilinear A
    """
    if not bridge_available():
        err = bridge_error() or "unknown import error"
        raise RuntimeError(
            f"MedRS bridge hard-fail: medrs_patch_bridge is unavailable ({err})."
        )
    patches_a, patches_b = _bridge.sample_asymmetric_patches_fused(  # type: ignore[union-attr]
        np.asarray(volume_xyz, dtype=np.float32),
        np.asarray(affine_inv, dtype=np.float32),
        np.asarray(centers_a_world, dtype=np.float32),
        np.asarray(centers_b_world, dtype=np.float32),
        np.asarray(rotation_matrix, dtype=np.float32),
        float(window_a_wc),
        float(window_a_ww),
        float(window_b_wc),
        float(window_b_ww),
        bool(a_native_no_interp),
    )
    return (
        np.asarray(patches_a, dtype=np.float32),
        np.asarray(patches_b, dtype=np.float32),
    )
