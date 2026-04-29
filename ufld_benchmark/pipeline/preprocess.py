from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    width: int
    height: int
    color: str = "rgb"
    scale: float = 1.0 / 255.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


def preprocess_frame(frame_bgr: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (config.width, config.height), interpolation=cv2.INTER_LINEAR)
    if config.color == "rgb":
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif config.color == "bgr":
        image = resized
    else:
        raise ValueError(f"Unsupported color mode: {config.color}")

    tensor = image.astype(np.float32) * config.scale
    mean = np.asarray(config.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config.std, dtype=np.float32).reshape(1, 1, 3)
    tensor = (tensor - mean) / std
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0).astype(np.float32, copy=False)
