from __future__ import annotations

import numpy as np

from volume_benchmarking.rust_bridge import sample_patches_trilinear


def test_rust_bridge_fallback_shape() -> None:
    vol = np.arange(16 * 16 * 8, dtype=np.float32).reshape(16, 16, 8)
    coords = np.zeros((2, 16, 16, 3), dtype=np.float32)
    coords[..., 0] = 4.0
    coords[..., 1] = 5.0
    coords[..., 2] = 2.0
    out = sample_patches_trilinear(vol, coords)
    assert out.shape == (2, 16, 16)
