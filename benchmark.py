from __future__ import annotations

import argparse
import json
from pathlib import Path

from ufld_benchmark.pipeline.postprocess import PostprocessConfig
from ufld_benchmark.pipeline.preprocess import PreprocessConfig
from ufld_benchmark.runner import BenchmarkConfig, run_benchmark


def main() -> int:
    args = parse_args()
    file_config = load_config(args.config)

    preprocess_config = PreprocessConfig(
        width=args.width if args.width is not None else file_config.get("width", 1600),
        height=args.height if args.height is not None else file_config.get("height", 320),
        color=file_config.get("color", "rgb"),
        scale=file_config.get("scale", 1.0 / 255.0),
        mean=tuple(file_config.get("mean", [0.485, 0.456, 0.406])),
        std=tuple(file_config.get("std", [0.229, 0.224, 0.225])),
    )

    config = BenchmarkConfig(
        model_path=Path(args.model),
        input_path=Path(args.input),
        runtime=args.runtime,
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        preprocess=preprocess_config,
        postprocess=PostprocessConfig(mode=file_config.get("postprocess", "summary")),
        warmup=args.warmup,
        max_frames=args.max_frames,
        intra_op_num_threads=args.intra_op_num_threads,
    )
    result = run_benchmark(config)
    summary = result["summary"]
    print(json.dumps(summary, indent=2))
    print(f"CSV: {config.output_csv}")
    print(f"JSON: {config.output_json}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFLD-v2 D3-G ONNX Runtime CPU benchmark")
    parser.add_argument("--model", required=True, help="Path to UFLD-v2 ONNX model")
    parser.add_argument("--input", required=True, help="Image, image directory, or video path")
    parser.add_argument("--runtime", default="onnxruntime_cpu", help="Runtime backend. Phase 1: onnxruntime_cpu only")
    parser.add_argument("--config", default="configs/ufldv2_onnx_cpu.json", help="JSON preprocessing config")
    parser.add_argument("--width", type=int, default=None, help="Override model input width")
    parser.add_argument("--height", type=int, default=None, help="Override model input height")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup frames excluded from results")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to read")
    parser.add_argument("--intra-op-num-threads", type=int, default=None, help="ONNX Runtime intra-op thread count")
    parser.add_argument("--output-csv", default="results/benchmark.csv", help="CSV result path")
    parser.add_argument("--output-json", default="results/benchmark.json", help="JSON summary path")
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
