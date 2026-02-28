"""CLI entrypoint for benchmark runs and summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from volume_benchmarking.runner import run_cell_from_cli, run_matrix_benchmark, summarize_raw_metrics


def _parse_backends(raw: str) -> list[str] | None:
    text = raw.strip().lower()
    if text in {"", "all"}:
        return None
    return [v.strip() for v in raw.split(",") if v.strip()]


def _print_result(label: str, payload: dict) -> None:
    pretty = {k: str(v) for k, v in payload.items()}
    print(f"{label}: {json.dumps(pretty, indent=2, sort_keys=True)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vb", description="Volume dataloader benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="Benchmark operations")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)

    run = bench_sub.add_parser("run", help="Run full matrix")
    run.add_argument("--matrix", required=True, help="Path to matrix yaml")
    run.add_argument("--runtime", default=None, help="Path to runtime yaml")
    run.add_argument("--catalog", default=None, help="Override catalog path")
    run.add_argument(
        "--backends",
        default="all",
        help="Comma-separated backend list or 'all'",
    )

    run_cell = bench_sub.add_parser("run-cell", help="Run one benchmark cell")
    run_cell.add_argument("--backend", required=True, choices=["custom_fused", "monai", "medrs"])
    run_cell.add_argument("--cache-state", required=True, choices=["cold_approx", "warm_pool"])
    run_cell.add_argument("--workers", type=int, required=True)
    run_cell.add_argument("--n-patches", type=int, required=True)
    run_cell.add_argument("--batch-size", type=int, required=True)
    run_cell.add_argument("--catalog", required=True)
    run_cell.add_argument("--runtime", default=None, help="Path to runtime yaml")
    run_cell.add_argument("--warmup-batches", type=int, default=10)
    run_cell.add_argument("--timed-batches", type=int, default=50)

    summarize = bench_sub.add_parser("summarize", help="Summarize existing raw JSONL files")
    summarize.add_argument("--input", nargs="+", required=True, help="One or more glob patterns")
    summarize.add_argument("--output-root", default=".", help="Output root for summary artifacts")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "bench" and args.bench_command == "run":
        payload = run_matrix_benchmark(
            matrix_path=args.matrix,
            runtime_path=args.runtime,
            backends=_parse_backends(args.backends),
            catalog_override=args.catalog,
        )
        _print_result("run", payload)
        return 0

    if args.command == "bench" and args.bench_command == "run-cell":
        payload = run_cell_from_cli(
            backend=args.backend,
            cache_state=args.cache_state,
            workers=args.workers,
            n_patches=args.n_patches,
            batch_size=args.batch_size,
            catalog=args.catalog,
            runtime_path=args.runtime,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        )
        _print_result("run-cell", payload)
        return 0

    if args.command == "bench" and args.bench_command == "summarize":
        csv_path, md_path = summarize_raw_metrics(args.input, output_root=args.output_root)
        _print_result("summarize", {"summary_csv": csv_path, "summary_md": md_path})
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
