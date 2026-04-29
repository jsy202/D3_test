from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_input_frames(input_path: str | Path) -> Iterator[tuple[int, str, np.ndarray]]:
    path = Path(input_path)
    if path.is_dir():
        yield from _iter_image_dir(path)
        return

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        yield 0, str(path), frame
        return

    if suffix in VIDEO_EXTENSIONS:
        yield from _iter_video(path)
        return

    raise ValueError(f"Unsupported input path: {path}")


def _iter_image_dir(path: Path) -> Iterator[tuple[int, str, np.ndarray]]:
    images = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images found in directory: {path}")

    for index, image_path in enumerate(images):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        yield index, str(image_path), frame


def _iter_video(path: Path) -> Iterator[tuple[int, str, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, str(path), frame
            index += 1
    finally:
        capture.release()
