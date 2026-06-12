#!/usr/bin/env bash
set -euo pipefail

uv run meld-emotion compare --config configs/example_suite.yaml
uv run meld-emotion status

uv run meld-emotion run --config configs/meld_sequence_dialogue_rnn.yaml
uv run meld-emotion compare --config configs/meld_embeddinggemma_wav2vec2_suite.yaml
uv run meld-emotion run --config configs/example_finegrained_xai.yaml
