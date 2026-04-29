# UFLD-v2 D3-G 벤치마크

이 프로젝트는 D3-G 보드에서 UFLD-v2 ONNX 모델을 CPU로 실행했을 때 실사용 가능한지 측정하기 위한 벤치마크 도구입니다.

1차 구현에서 실제로 사용하는 것은 아래 항목뿐입니다.

- ONNX 모델 포맷
- ONNX Runtime CPU execution provider
- OpenCV 프레임/이미지 로딩 및 전처리
- NumPy 텐서 처리
- CSV/JSON 결과 저장

D3-G 보드에서는 아래 항목을 사용하지 않습니다.

- PyTorch
- TensorRT
- CUDA
- `onnxruntime-gpu`

후속 후보로만 남겨둔 항목은 아래와 같습니다.

- TNN OpenCL
- ncnn Vulkan
- MNN

코드에는 후속 런타임 후보에 대한 TODO placeholder만 있으며, 현재 실제 동작하는 런타임은 `onnxruntime_cpu` 하나입니다.

## 지연시간 정의

이 벤치마크에서 판단 기준으로 쓰는 지연시간은 하나입니다.

```text
latency_ms = preprocess_ms + inference_ms + postprocess_ms
```

`latency_ms`에 포함하지 않는 항목은 아래와 같습니다.

- 비디오 프레임 read
- 카메라 capture
- 이미지 파일 read
- `cv2.imshow`
- CSV write
- JSON write
- 모델 loading

다만 원인 분석을 위해 `preprocess_ms`, `inference_ms`, `postprocess_ms`는 CSV에 따로 저장합니다.

## 코드상 실제 측정 구간

실제 측정 구간은 `ufld_benchmark/runner.py` 안에 있습니다.

프레임 하나마다 벤치마크는 아래 순서로 동작합니다.

```text
frame read                  -> 측정 제외
preprocess_frame(...)       -> preprocess_ms
runtime.infer(...)          -> inference_ms
postprocess_outputs(...)    -> postprocess_ms
CSV/JSON write              -> 측정 제외
```

즉 `latency_ms`는 이미 메모리에 올라온 프레임 한 장을 받아서, 전처리하고, ONNX Runtime CPU로 추론하고, 출력 결과를 요약하는 데 걸린 CPU 측 시간입니다.

```text
latency_ms = preprocess_ms + inference_ms + postprocess_ms
```

이 값으로 D3-G에서 UFLD-v2가 목표 FPS를 따라갈 수 있는지 판단합니다. 예를 들어 30 FPS가 목표라면 프레임당 약 `33.3 ms` 이하가 필요하고, 15 FPS가 목표라면 프레임당 약 `66.7 ms` 이하가 필요합니다.

CSV에는 세부 구간도 따로 남기므로, 시간이 OpenCV 전처리에서 많이 쓰이는지, ONNX Runtime 추론에서 많이 쓰이는지, 후처리에서 많이 쓰이는지 확인할 수 있습니다.

## 로컬 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

로컬 smoke test까지 실행하려면 개발용 의존성도 설치합니다.

```bash
pip install -r requirements-dev.txt
```

## Smoke Test

smoke test는 작은 identity ONNX 모델과 synthetic lane 이미지를 생성한 뒤 벤치마크 도구가 정상 동작하는지 확인합니다.

이 테스트는 UFLD-v2 정확도나 실제 성능을 측정하는 용도가 아닙니다.

```bash
bash scripts/run_smoke.sh
```

출력 파일:

- `results/smoke.csv`
- `results/smoke.json`

## UFLD-v2 ONNX로 실행

ONNX 모델을 `models/` 아래에 둡니다.

예:

```text
models/ufldv2_culane_res18_320x1600.onnx
```

이미지, 이미지 디렉터리, 비디오 파일을 입력으로 사용할 수 있습니다.

```bash
python3 benchmark.py \
  --model models/ufldv2_culane_res18_320x1600.onnx \
  --input example.mp4 \
  --runtime onnxruntime_cpu \
  --config configs/ufldv2_onnx_cpu.json \
  --warmup 10 \
  --max-frames 300 \
  --output-csv results/d3g_ufldv2.csv \
  --output-json results/d3g_ufldv2.json
```

공식 UFLD-v2 저장소에는 `example.mp4`가 포함되어 있으므로 첫 end-to-end 실행용으로 사용할 수 있습니다.

## D3-G 보드에서 바로 실행

D3-G 보드에서 이 저장소를 clone한 뒤 아래 명령을 실행합니다.

```bash
git clone https://github.com/jsy202/D3_test.git
cd D3_test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models inputs results
```

export된 UFLD-v2 ONNX 모델은 `models/`에 넣고, 테스트할 비디오나 이미지는 `inputs/`에 넣습니다.

비디오 벤치마크 실행:

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

단일 이미지 벤치마크 실행:

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

만약 보드에서 `python3 -m venv .venv`가 실패하면 `python3-venv`가 설치되어 있지 않은 상태일 수 있습니다.

그 경우 먼저 아래 명령을 실행합니다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

Docker는 보드의 Python, OpenCV, ONNX Runtime 설치 경로가 안정화된 뒤에 환경 고정용으로 사용하는 것을 권장합니다.

## 메모

- 벤치마크는 `CPUExecutionProvider`를 강제합니다.
- 모델 로딩 시간은 측정 루프 밖에 있습니다.
- 프레임 read는 전처리 타이머가 시작되기 전에 수행됩니다.
- warmup 프레임은 실제 실행은 하지만 CSV/JSON summary 통계에서는 제외됩니다.
