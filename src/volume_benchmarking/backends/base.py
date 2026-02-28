"""Base interface for benchmark backend adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from volume_benchmarking.contracts import BenchmarkCellConfig, BenchmarkRuntimeConfig, VolumeRecord


@dataclass(frozen=True)
class CellContext:
    """Resolved cell + runtime inputs provided to every backend."""

    cell: BenchmarkCellConfig
    runtime: BenchmarkRuntimeConfig
    records: list[VolumeRecord]


class BackendAdapter(ABC):
    """Every backend must prepare/build/teardown with a consistent API."""

    def __init__(self, *, records: list[VolumeRecord], runtime: BenchmarkRuntimeConfig) -> None:
        self.records = records
        self.runtime = runtime

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def prepare(self, cell_ctx: CellContext) -> None:
        pass

    @abstractmethod
    def build_dataset(self, cell_ctx: CellContext) -> Iterable[dict[str, Any]]:
        pass

    @abstractmethod
    def teardown(self) -> None:
        pass
