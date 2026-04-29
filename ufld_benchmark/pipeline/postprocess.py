from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PostprocessConfig:
    mode: str = "summary"


def postprocess_outputs(outputs: list[np.ndarray], config: PostprocessConfig) -> dict[str, Any]:
    if config.mode != "summary":
        raise ValueError(f"Unsupported postprocess mode: {config.mode}")

    return {
        "num_outputs": len(outputs),
        "output_shapes": [list(output.shape) for output in outputs],
        "output_dtypes": [str(output.dtype) for output in outputs],
    }
