from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ufld_benchmark.runtimes.base import RuntimeBackend


class OnnxRuntimeCpuBackend(RuntimeBackend):
    name = "onnxruntime_cpu"

    def __init__(self, model_path: str | Path, intra_op_num_threads: int | None = None) -> None:
        session_options = ort.SessionOptions()
        if intra_op_num_threads is not None:
            session_options.intra_op_num_threads = intra_op_num_threads

        self.model_path = str(model_path)
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input = self.session.get_inputs()[0]
        self._output_names = [output.name for output in self.session.get_outputs()]

    @property
    def input_name(self) -> str:
        return self._input.name

    @property
    def input_shape(self) -> tuple[int | None, ...]:
        return tuple(None if isinstance(dim, str) else dim for dim in self._input.shape)

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        outputs = self.session.run(self._output_names, {self.input_name: tensor})
        return [np.asarray(output) for output in outputs]

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "model_path": self.model_path,
            "providers": self.session.get_providers(),
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "outputs": self._output_names,
        }
