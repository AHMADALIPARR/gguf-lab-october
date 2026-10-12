#!/usr/bin/env bash
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
#
# Pull the October 2026 GGUF drops. Run on the TARGET machine
# (this VM cannot hold these files). Requires: huggingface-cli
# (pip install huggingface_hub) or ollama.
#
# Usage: ./pull.sh [all|embed|mellum|qwen27|bonsai|xing|flash]
set -euo pipefail

STORE="${GGUF_STORE:-$HOME/gguf-lab}"
mkdir -p "$STORE"

pull() { # repo, file, dest
  echo "== $1 / $2"
  huggingface-cli download "$1" "$2" --local-dir "$STORE/$3" --local-dir-use-symlinks False
}

TARGET="${1:-all}"

case "$TARGET" in
  all|embed)
    pull unsloth/embeddinggemma-2-GGUF "embeddinggemma-2-Q4_K_M.gguf" embed ;;
esac
case "$TARGET" in
  all|mellum)
    pull JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF \
      "Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf" mellum ;;
esac
case "$TARGET" in
  all|qwen27)
    pull unsloth/Qwen3.8-27B-GGUF "Qwen3.8-27B-UD-Q4_K_M.gguf" qwen27 ;;
esac
case "$TARGET" in
  all|bonsai)
    pull prism-ml/Ternary-Bonsai-2-27B-gguf "Ternary-Bonsai-2-27B-PTQ1_0.gguf" bonsai ;;
esac
case "$TARGET" in
  all|xing)
    pull XingChen-AGI/Xing4.0-29B-A4B-GGUF "xing4_0-29b-IQ4_NL.gguf" xing ;;
esac
case "$TARGET" in
  all|flash)
    echo "Flash Next 177B: offload-only; pull manually when ready:"
    echo "  huggingface-cli download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --local-dir $STORE/flash" ;;
esac

echo "Done. Store: $STORE"
du -sh "$STORE"/* 2>/dev/null || true
