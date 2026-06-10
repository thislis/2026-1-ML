#!/usr/bin/env bash
set -euo pipefail

uv run meld-emotion compare --config configs/all_model_w_all_features.yaml
uv run meld-emotion status

uv run meld-emotion run --config configs/conformer_sequence_dialogue_rnn.yaml
uv run meld-emotion run --config configs/conformer_sequence_fused_only.yaml
uv run meld-emotion run --config configs/conformer_sequence_fused_context.yaml
uv run meld-emotion run --config configs/conformer_sequence_fused_memory.yaml
uv run meld-emotion run --config configs/conformer_sequence_full.yaml
uv run meld-emotion run --config configs/conformer_sequence_imbalance_calibrated.yaml
uv run meld-emotion run --config configs/conformer_sequence_neutral_gate.yaml
uv run meld-emotion run --config configs/conformer_sequence_ensemble.yaml
uv run meld-emotion run --config configs/conformer_sequence_moe_rare.yaml
