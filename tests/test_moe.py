"""Mixture-of-experts classifier."""

from __future__ import annotations

import numpy as np

from meld_emotion.config.loader import from_dict
from meld_emotion.config.schema import MoeSettings, RareExpertSettings
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.models.moe import MoeEmotionClassifier


def test_moe_classifier_predicts_and_records_routing_metadata(
    train_bundle,
    test_bundle,
    y_train,
) -> None:
    settings = MoeSettings(
        top_k=2,
        class_aware_routing=True,
        rare_expert=RareExpertSettings(target_classes=(5, 6)),
    )
    clf = MoeEmotionClassifier(settings, EmotionLabelEncoder().classes)

    clf.fit(train_bundle, y_train)
    prediction = clf.predict(test_bundle)

    assert prediction.proba.shape == (test_bundle.n_samples, 7)
    assert np.allclose(prediction.proba.sum(axis=1), 1.0)
    stats = clf.last_gate_stats
    assert "load_balancing_loss" in stats
    assert "moe_gate_rare_mean" in stats
    assert "moe_selected_rare_rate" in stats


def test_moe_config_loads() -> None:
    cfg = from_dict(
        {
            "name": "conformer_sequence_moe_rare",
            "dataset": {"type": "synthetic", "n_train": 24, "n_dev": 0, "n_test": 12},
            "extractors": [
                {"type": "text_concepts"},
                {"type": "audio_concepts"},
                {"type": "video_concepts"},
            ],
            "model": {
                "type": "moe",
                "moe": {
                    "routing": "top2",
                    "top_k": 2,
                    "class_aware_routing": True,
                    "rare_expert": {
                        "enabled": True,
                        "target_classes": [5, 6],
                        "loss_weight": 0.5,
                        "hard_negative_weight": 1.5,
                    },
                },
            },
            "reporters": [],
        }
    )
    assert cfg.model.type == "moe"
    assert cfg.model.moe.rare_expert.target_classes == (5, 6)
