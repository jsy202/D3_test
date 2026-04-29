# UFLD-v2 D3-G Benchmark

This project measures whether UFLD-v2 ONNX inference is practical on a D3-G board.

Phase 1 intentionally runs only:

- ONNX model format
- ONNX Runtime CPU execution provider
- OpenCV frame/image loading and preprocessing
- NumPy tensor handling
- CSV/JSON result output

Not used on the board:

- PyTorch
- TensorRT
- CUDA
- `onnxruntime-gpu`

Future candidates only:

- TNN OpenCL
- ncnn Vulkan
- MNN

The code contains TODO runtime placeholders for those candidates, but the only implemented runtime is `onnxruntime_cpu`.

## Latency Definition

The benchmark reports one decision latency:

```text
latency_ms = preprocess_ms + inference_ms + postprocess_ms
```

Excluded from `latency_ms`:

- video read
- camera capture
- image file read
- `cv2.imshow`
- CSV write
- JSON write
- model loading

Per-stage timings are still written to CSV for diagnosis.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the local smoke test, install dev dependencies too:

```bash
pip install -r requirements-dev.txt
```

## Smoke Test

The smoke test creates a tiny identity ONNX model and a synthetic lane image. It verifies the benchmark tool, not UFLD-v2 accuracy.

```bash
bash scripts/run_smoke.sh
```

Outputs:

- `results/smoke.csv`
- `results/smoke.json`

## Run With UFLD-v2 ONNX

Place the ONNX model under `models/`, for example:

```text
models/ufldv2_culane_res18_320x1600.onnx
```

Run on an image, image directory, or video:

```bash
python benchmark.py \
  --model models/ufldv2_culane_res18_320x1600.onnx \
  --input example.mp4 \
  --runtime onnxruntime_cpu \
  --config configs/ufldv2_onnx_cpu.json \
  --warmup 10 \
  --max-frames 300 \
  --output-csv results/d3g_ufldv2.csv \
  --output-json results/d3g_ufldv2.json
```

The official UFLD-v2 repository includes `example.mp4`, which is useful for a first end-to-end run.

## D3-G Transfer Workflow

Recommended first workflow:

```bash
git clone <repo-url>
cd ufldv2-d3g-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py --model models/<model>.onnx --input <input.mp4>
```

Use Docker only after the board environment is known and the Python path is stable.

## Notes

- The benchmark forces `CPUExecutionProvider`.
- Model loading is outside the measured loop.
- Frame read happens before preprocess timing starts.
- Warmup frames are executed but excluded from CSV/JSON summary metrics.
