#!/usr/bin/env bash
set -euo pipefail

python3 scripts/create_smoke_assets.py
python3 benchmark.py \
  --model models/smoke_identity.onnx \
  --input samples/smoke_lane.jpg \
  --width 1600 \
  --height 320 \
  --warmup 0 \
  --max-frames 1 \
  --output-csv results/smoke.csv \
  --output-json results/smoke.json
