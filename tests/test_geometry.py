from __future__ import annotations

import numpy as np

from volume_benchmarking.geometry import euler_xyz_to_matrix, patch_plane_offsets_mm


def test_euler_matrix_is_orthonormal() -> None:
    rot = euler_xyz_to_matrix((10.0, -15.0, 33.0))
    ident = rot @ rot.T
    np.testing.assert_allclose(ident, np.eye(3, dtype=np.float32), atol=1e-5)


def test_patch_plane_offsets_shape() -> None:
    offsets = patch_plane_offsets_mm(size=16)
    assert offsets.shape == (16, 16, 3)
    assert np.isfinite(offsets).all()
