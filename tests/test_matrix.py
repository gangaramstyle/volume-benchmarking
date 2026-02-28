from __future__ import annotations

from volume_benchmarking.matrix import expand_cells, load_matrix_config


def test_default_matrix_cell_count() -> None:
    cfg = load_matrix_config("configs/matrix.default.yaml")
    cells = expand_cells(cfg)
    assert len(cells) == 144
