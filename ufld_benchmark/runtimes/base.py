from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RuntimeBackend(ABC):
    """Minimal runtime contract used by the benchmark runner."""

    name: str

    @property
    @abstractmethod
    def input_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int | None, ...]:
        raise NotImplementedError

    @abstractmethod
    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"runtime": self.name}
