from __future__ import annotations

import numpy as np

from volume_benchmarking.contracts import VolumeRecord
from volume_benchmarking.runner import _select_record, _validate_sample_contract


def test_select_record_by_index() -> None:
    records = [
        VolumeRecord(scan_id="s0", nifti_path="/tmp/a.nii", modality="CT"),
        VolumeRecord(scan_id="s1", nifti_path="/tmp/b.nii", modality="MR"),
    ]
    rec = _select_record(records, record_index=1, scan_id=None)
    assert rec.scan_id == "s1"


def test_select_record_by_scan_id() -> None:
    records = [
        VolumeRecord(scan_id="s0", nifti_path="/tmp/a.nii", modality="CT"),
        VolumeRecord(scan_id="s1", nifti_path="/tmp/b.nii", modality="MR"),
    ]
    rec = _select_record(records, record_index=0, scan_id="s1")
    assert rec.scan_id == "s1"


def test_validate_sample_contract_shapes() -> None:
    sample = {
        "patches_a": np.zeros((8, 1, 16, 16), dtype=np.float32),
        "patches_b": np.zeros((8, 1, 16, 16), dtype=np.float32),
        "rel_coords_a_mm": np.zeros((8, 3), dtype=np.float32),
        "rel_coords_b_mm": np.zeros((8, 3), dtype=np.float32),
        "anchor_delta_mm_a_frame": np.zeros((3,), dtype=np.float32),
        "rotation_b_euler_deg": np.zeros((3,), dtype=np.float32),
        "rotation_b_matrix": np.zeros((3, 3), dtype=np.float32),
        "meta": {},
    }
    checks = _validate_sample_contract(sample, n_patches=8)
    assert checks["has_all_required_keys"] is True
    assert checks["all_shapes_match"] is True
