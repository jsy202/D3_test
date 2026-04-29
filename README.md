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

## What Is Timed In Code

The measured section is inside `ufld_benchmark/runner.py`.

For each frame, the benchmark does this:

```text
frame read                  -> not timed
preprocess_frame(...)       -> preprocess_ms
runtime.infer(...)          -> inference_ms
postprocess_outputs(...)    -> postprocess_ms
CSV/JSON write              -> not timed
```

`latency_ms` means the CPU-side time needed after a frame is already available in memory and before the result has been summarized:

```text
latency_ms = preprocess_ms + inference_ms + postprocess_ms
```

This is the value to use when deciding whether UFLD-v2 can keep up with a target frame rate on D3-G. For example, 30 FPS needs about `33.3 ms` or less per frame, and 15 FPS needs about `66.7 ms` or less per frame.

The benchmark also stores `preprocess_ms`, `inference_ms`, and `postprocess_ms` separately so it is clear whether time is being spent in OpenCV preprocessing, ONNX Runtime inference, or output handling.

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

Recommended first workflow after cloning on the D3-G board:

```bash
git clone https://github.com/jsy202/D3_test.git
cd D3_test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models inputs results
```

Copy the exported UFLD-v2 ONNX model into `models/` and a test video or image into `inputs/`.

Run a video benchmark:

```bash
python3 benchmark.py \
  --model models/ufldv2_culane_res18_320x1600.onnx \
  --input inputs/example.mp4 \
  --runtime onnxruntime_cpu \
  --config configs/ufldv2_onnx_cpu.json \
  --warmup 10 \
  --max-frames 300 \
  --output-csv results/d3g_ufldv2.csv \
  --output-json results/d3g_ufldv2.json
```

Run a single image benchmark:

```bash
python3 benchmark.py \
  --model models/ufldv2_culane_res18_320x1600.onnx \
  --input inputs/frame.jpg \
  --runtime onnxruntime_cpu \
  --config configs/ufldv2_onnx_cpu.json \
  --warmup 0 \
  --max-frames 1 \
  --output-csv results/d3g_image.csv \
  --output-json results/d3g_image.json
```

If `python3 -m venv .venv` fails because `python3-venv` is not installed on the board, install it first:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

Use Docker only after the board environment is known and the Python path is stable.

## Notes

- The benchmark forces `CPUExecutionProvider`.
- Model loading is outside the measured loop.
- Frame read happens before preprocess timing starts.
- Warmup frames are executed but excluded from CSV/JSON summary metrics.
