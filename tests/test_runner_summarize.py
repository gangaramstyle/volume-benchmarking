from __future__ import annotations

import json
from pathlib import Path

from volume_benchmarking.runner import summarize_raw_metrics


def test_summarize_from_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "cell_summary",
                        "run_id": "r1",
                        "cell_id": "c1",
                        "backend": "custom_fused",
                        "cache_state": "warm_pool",
                        "workers": 0,
                        "n_patches": 16,
                        "batch_size": 4,
                        "warmup_batches": 1,
                        "timed_batches": 2,
                        "time_to_first_batch_ms": 10.0,
                        "throughput_samples_per_sec": 2.0,
                        "timed_wall_seconds": 2.0,
                        "samples_timed": 4,
                        "peak_cpu_rss_gb_global": 1.0,
                        "peak_vram_allocated_gb": 0.0,
                        "peak_vram_reserved_gb": 0.0,
                        "gpu_starvation_wait_ms_total": 5.0,
                        "gpu_starvation_wait_ms_mean": 2.5,
                        "replacement_wait_ms_total": 0.0,
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path, md_path = summarize_raw_metrics([str(raw)], output_root=str(tmp_path))
    assert csv_path.exists()
    assert md_path.exists()
