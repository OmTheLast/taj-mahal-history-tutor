#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-taj-step100-Q6_K.gguf}"
llama-cli \
  -m "$MODEL_PATH" \
  -p "Where is the Taj Mahal?" \
  -st \
  -n 128 \
  --temp 0

