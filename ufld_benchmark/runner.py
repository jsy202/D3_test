from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ufld_benchmark.pipeline.inputs import iter_input_frames
from ufld_benchmark.pipeline.postprocess import PostprocessConfig, postprocess_outputs
from ufld_benchmark.pipeline.preprocess import PreprocessConfig, preprocess_frame
from ufld_benchmark.pipeline.results import summarize, write_csv, write_json
from ufld_benchmark.runtimes.factory import create_runtime


@dataclass(frozen=True)
class BenchmarkConfig:
    model_path: Path
    input_path: Path
    runtime: str
    output_csv: Path
    output_json: Path
    preprocess: PreprocessConfig
    postprocess: PostprocessConfig
    warmup: int = 5
    max_frames: int | None = None
    intra_op_num_threads: int | None = None


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    runtime = create_runtime(
        config.runtime,
        model_path=config.model_path,
        intra_op_num_threads=config.intra_op_num_threads,
    )

    rows: list[dict[str, Any]] = []
    last_postprocess: dict[str, Any] | None = None
    total_seen = 0

    for frame_index, source, frame in iter_input_frames(config.input_path):
        if config.max_frames is not None and total_seen >= config.max_frames:
            break
        total_seen += 1

        preprocess_start = time.perf_counter()
        tensor = preprocess_frame(frame, config.preprocess)
        preprocess_ms = _elapsed_ms(preprocess_start)

        inference_start = time.perf_counter()
        outputs = runtime.infer(tensor)
        inference_ms = _elapsed_ms(inference_start)

        postprocess_start = time.perf_counter()
        last_postprocess = postprocess_outputs(outputs, config.postprocess)
        postprocess_ms = _elapsed_ms(postprocess_start)

        if frame_index < config.warmup:
            continue

        latency_ms = preprocess_ms + inference_ms + postprocess_ms
        rows.append(
            {
                "frame_index": frame_index,
                "source": source,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "latency_ms": latency_ms,
            }
        )

    summary = summarize(rows)
    payload = {
        "summary": summary,
        "config": _config_payload(config),
        "runtime": runtime.metadata(),
        "warmup_frames": config.warmup,
        "measured_frames": len(rows),
        "seen_frames": total_seen,
        "last_postprocess": last_postprocess,
        "latency_definition": "latency_ms = preprocess_ms + inference_ms + postprocess_ms",
        "latency_excludes": [
            "video read",
            "camera capture",
            "image file read",
            "cv2.imshow",
            "CSV write",
            "JSON write",
            "model loading",
        ],
    }

    write_csv(config.output_csv, rows)
    write_json(config.output_json, payload)
    return payload


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _config_payload(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "model_path": str(config.model_path),
        "input_path": str(config.input_path),
        "runtime": config.runtime,
        "output_csv": str(config.output_csv),
        "output_json": str(config.output_json),
        "preprocess": {
            "width": config.preprocess.width,
            "height": config.preprocess.height,
            "color": config.preprocess.color,
            "scale": config.preprocess.scale,
            "mean": list(config.preprocess.mean),
            "std": list(config.preprocess.std),
        },
        "postprocess": {"mode": config.postprocess.mode},
        "warmup": config.warmup,
        "max_frames": config.max_frames,
        "intra_op_num_threads": config.intra_op_num_threads,
    }
