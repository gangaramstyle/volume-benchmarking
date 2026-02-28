from __future__ import annotations

import numpy as np
import pytest

from volume_benchmarking.geometry import VolumeGeometry
from volume_benchmarking.payload import VolumeContext, build_asymmetric_sample


def test_payload_contract_shapes() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    vol = np.random.randn(64, 64, 32).astype(np.float32)
    affine = np.eye(4, dtype=np.float32)
    inv = np.linalg.inv(affine).astype(np.float32)
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ctx = VolumeContext(
        scan_id="synthetic",
        volume_xyz=vol,
        affine=affine,
        affine_inv=inv,
        spacing_mm=spacing,
        geometry=VolumeGeometry(
            shape_xyz=(64, 64, 32),
            spacing_mm=(1.0, 1.0, 1.0),
            affine=affine,
            affine_inv=inv,
        ),
    )

    sample = build_asymmetric_sample(
        ctx=ctx,
        n_patches=8,
        device="cpu",
        rng=rng,
        backend="custom_fused",
        cache_state="warm_pool",
        worker_id=0,
    )

    assert sample["patches_a"].shape == (8, 1, 16, 16)
    assert sample["patches_b"].shape == (8, 1, 16, 16)
    assert sample["rel_coords_a_mm"].shape == (8, 3)
    assert sample["rel_coords_b_mm"].shape == (8, 3)
    assert sample["anchor_delta_mm_a_frame"].shape == (3,)
    assert sample["rotation_b_euler_deg"].shape == (3,)
    assert sample["rotation_b_matrix"].shape == (3, 3)
    assert isinstance(sample["meta"], dict)
    assert sample["patches_a"].dtype == torch.float32
