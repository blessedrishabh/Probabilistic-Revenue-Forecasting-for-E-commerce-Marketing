#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

python Forecasting/main.py --data_dir "$DATA_DIR" --model_path "$MODEL_PATH" --output_path "$OUTPUT_PATH"