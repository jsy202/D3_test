from __future__ import annotations

from pathlib import Path

from ufld_benchmark.runtimes.onnxruntime_cpu import OnnxRuntimeCpuBackend


def create_runtime(name: str, model_path: str | Path, intra_op_num_threads: int | None = None):
    if name == "onnxruntime_cpu":
        return OnnxRuntimeCpuBackend(model_path=model_path, intra_op_num_threads=intra_op_num_threads)
    raise ValueError(f"Unsupported runtime '{name}'. Phase 1 supports only 'onnxruntime_cpu'.")
