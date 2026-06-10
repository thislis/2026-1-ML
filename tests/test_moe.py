"""Mixture-of-experts classifier."""

from __future__ import annotations

import numpy as np

from meld_emotion.config.loader import load_config
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
    cfg = load_config("configs/conformer_sequence_moe_rare.yaml")
    assert cfg.model.type == "moe"
    assert cfg.model.moe.rare_expert.target_classes == (5, 6)
