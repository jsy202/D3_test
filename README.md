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

## 처음부터 끝까지 실행 절차

아래 절차는 D3-G 보드에서 완전히 처음 시작한다고 가정합니다.

최종적으로 필요한 파일은 2개입니다.

```text
models/*.onnx          실제 UFLD-v2 ONNX 모델
inputs/example.mp4     테스트할 영상 또는 이미지
```

### 1. 벤치마크 repo 받기

```bash
git clone https://github.com/jsy202/D3_test.git
cd D3_test
mkdir -p models inputs results
```

### 2. Python 환경 만들기

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

만약 `python3 -m venv .venv`가 실패하면 `python3-venv`가 설치되어 있지 않은 상태일 수 있습니다.

그 경우 먼저 아래 명령을 실행한 뒤 다시 venv를 만듭니다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 도구 자체가 실행되는지 확인

이 단계는 실제 UFLD-v2 성능 측정이 아니라, 보드에서 ONNX Runtime CPU, OpenCV, NumPy, CSV/JSON 저장이 정상 동작하는지 확인하는 smoke test입니다.

```bash
bash scripts/run_smoke.sh
```

성공하면 아래 파일이 생깁니다.

```text
results/smoke.csv
results/smoke.json
```

여기에 나온 latency는 synthetic identity ONNX 모델 결과이므로 UFLD-v2 성능값으로 보면 안 됩니다.

### 4. 실제 UFLD-v2 ONNX 모델 준비

방법 A는 이미 export된 ONNX 모델을 받는 방식입니다. 내일 보드에서 바로 측정하려면 이 방법이 가장 단순합니다.

```bash
cd ..
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/324_Ultra-Fast-Lane-Detection-v2
bash download.sh
```

다운로드가 끝나면 ONNX 파일을 찾습니다.

```bash
find . -maxdepth 3 -name "*.onnx"
```

찾은 ONNX 파일을 `D3_test/models/`로 복사합니다.

예를 들어 파일명이 `ufldv2_culane_res34_320x1600.onnx`라면:

```bash
cp ./ufldv2_culane_res34_320x1600.onnx ../../D3_test/models/
cd ../../D3_test
```

파일이 하위 디렉터리에 있으면 `find` 결과에 나온 경로를 그대로 사용합니다.

예:

```bash
cp ./saved_model/ufldv2_culane_res34_320x1600.onnx ../../D3_test/models/
cd ../../D3_test
```

방법 B는 PC에서 PyTorch checkpoint를 ONNX로 export하는 방식입니다. 이 방법은 D3-G가 아니라 PC에서만 권장합니다.

```bash
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git
cd Ultra-Fast-Lane-Detection-v2
python deploy/pt2onnx.py \
  --config_path configs/culane_res34.py \
  --model_path weights/culane_res34.pth
```

생성된 ONNX 파일만 D3-G의 `D3_test/models/`로 복사합니다.

### 5. 테스트 영상 준비

직접 준비한 영상을 쓰는 경우:

```bash
cp /path/to/your/video.mp4 inputs/example.mp4
```

공식 UFLD-v2 예시 영상을 쓰는 경우:

```bash
cd ..
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git
cp Ultra-Fast-Lane-Detection-v2/example.mp4 D3_test/inputs/example.mp4
cd D3_test
```

### 6. 실제 UFLD-v2 벤치마크 실행

먼저 모델 파일명을 확인합니다.

```bash
ls models
```

모델 파일이 `ufldv2_culane_res34_320x1600.onnx`라면 아래처럼 실행합니다.

```bash
python3 benchmark.py \
  --model models/ufldv2_culane_res34_320x1600.onnx \
  --input inputs/example.mp4 \
  --runtime onnxruntime_cpu \
  --config configs/ufldv2_onnx_cpu.json \
  --warmup 10 \
  --max-frames 300 \
  --output-csv results/d3g_ufldv2.csv \
  --output-json results/d3g_ufldv2.json
```

모델 파일명이 다르면 `--model` 값만 실제 파일명에 맞게 바꿉니다.

예:

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

### 7. 결과 확인

실행이 끝나면 아래 파일을 확인합니다.

```text
results/d3g_ufldv2.csv
results/d3g_ufldv2.json
```

요약값만 빠르게 보려면:

```bash
cat results/d3g_ufldv2.json
```

중요하게 볼 값:

```text
latency_ms_avg          평균 지연시간
latency_ms_p95          느린 쪽 95% 지연시간
fps_from_latency_avg    평균 latency 기준 환산 FPS
```

실사용 가능성은 `latency_ms_avg`만 보지 말고 `latency_ms_p95`도 같이 봅니다. 평균은 괜찮아도 p95가 너무 크면 실제 영상에서는 끊김이 생길 수 있습니다.

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

## D3-G 보드에서 바로 실행 요약

이미 ONNX 모델과 입력 영상이 준비되어 있다면 아래 요약 명령만 사용해도 됩니다.

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

## 자주 나는 문제

`No such file or directory: models/...onnx`가 나오면 `--model` 경로와 실제 파일명이 다릅니다.

```bash
ls models
```

`Failed to open video`가 나오면 `--input` 경로가 잘못됐거나 OpenCV가 해당 영상 코덱을 읽지 못하는 상태입니다.

```bash
ls inputs
```

`No module named cv2` 또는 `No module named onnxruntime`가 나오면 venv가 활성화되지 않았거나 의존성 설치가 안 된 상태입니다.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`onnxruntime` 설치가 D3-G에서 실패하면 보드의 CPU 아키텍처와 Python 버전에 맞는 wheel이 없는 경우일 수 있습니다. 이 경우 Python 버전과 아키텍처를 먼저 확인합니다.

```bash
python3 --version
uname -m
```

## 메모

- 벤치마크는 `CPUExecutionProvider`를 강제합니다.
- 모델 로딩 시간은 측정 루프 밖에 있습니다.
- 프레임 read는 전처리 타이머가 시작되기 전에 수행됩니다.
- warmup 프레임은 실제 실행은 하지만 CSV/JSON summary 통계에서는 제외됩니다.
