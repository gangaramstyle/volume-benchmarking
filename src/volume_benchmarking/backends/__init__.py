"""Backend adapter registry."""

from volume_benchmarking.backends.base import BackendAdapter
from volume_benchmarking.backends.custom_fused import CustomFusedBackend
from volume_benchmarking.backends.medrs_backend import MedRSBackend
from volume_benchmarking.backends.monai_backend import MONAIBackend


def make_backend(name: str, **kwargs) -> BackendAdapter:
    lowered = str(name).strip().lower()
    if lowered == "custom_fused":
        return CustomFusedBackend(**kwargs)
    if lowered == "monai":
        return MONAIBackend(**kwargs)
    if lowered == "medrs":
        return MedRSBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'")


__all__ = ["BackendAdapter", "CustomFusedBackend", "MONAIBackend", "MedRSBackend", "make_backend"]
