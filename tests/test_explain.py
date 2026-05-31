"""설명기(permutation / modality ablation / counterfactual)."""

from __future__ import annotations

from meld_emotion.config.loader import from_dict, to_dict
from meld_emotion.config.schema import ExperimentConfig, PermutationConfig
from meld_emotion.core.types import FeatureKind
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.evaluation.metrics import MacroF1Metric
from meld_emotion.explain.counterfactual import CounterfactualExplainer
from meld_emotion.explain.modality_contribution import ModalityAblationExplainer
from meld_emotion.explain.permutation import PermutationImportanceExplainer
from meld_emotion.fusion.early import EarlyFusionClassifier
from meld_emotion.models.baselines import NearestCentroidEstimator


def _model(train_bundle, y_train):
    return EarlyFusionClassifier(NearestCentroidEstimator, EmotionLabelEncoder().classes).fit(
        train_bundle, y_train
    )


def test_permutation_returns_top_k(train_bundle, test_bundle, y_train, y_test) -> None:
    model = _model(train_bundle, y_train)
    report = PermutationImportanceExplainer(
        MacroF1Metric(), n_repeats=2, top_k=5, kinds=(FeatureKind.CONCEPT, FeatureKind.EMBEDDING)
    ).explain(model, test_bundle, y_test)
    assert len(report.feature_contributions) <= 5
    # 내림차순 정렬 확인
    importances = [c.importance for c in report.feature_contributions]
    assert importances == sorted(importances, reverse=True)


def test_permutation_default_targets_concepts_only(
    train_bundle, test_bundle, y_train, y_test
) -> None:
    # 기본값(kinds=concept)에서는 임베딩(bow_*) 특징이 대상에서 제외되어야 한다.
    model = _model(train_bundle, y_train)
    report = PermutationImportanceExplainer(MacroF1Metric(), n_repeats=2, top_k=50).explain(
        model, test_bundle, y_test
    )
    assert report.feature_contributions  # 개념 특징은 존재
    assert all(not c.name.startswith("bow_") for c in report.feature_contributions)


def test_permutation_can_include_embeddings(train_bundle, test_bundle, y_train, y_test) -> None:
    model = _model(train_bundle, y_train)
    report = PermutationImportanceExplainer(
        MacroF1Metric(),
        n_repeats=2,
        top_k=200,  # 전체 특징(<200)을 모두 후보로 포함
        kinds=(FeatureKind.CONCEPT, FeatureKind.EMBEDDING),
    ).explain(model, test_bundle, y_test)
    names = {c.name for c in report.feature_contributions}
    assert any(n.startswith("bow_") for n in names)  # 임베딩 특징이 후보에 포함됨


def test_permutation_config_kinds_roundtrip() -> None:
    config = ExperimentConfig(
        explainers=(PermutationConfig(kinds=("concept", "embedding")),)
    )
    assert from_dict(to_dict(config)) == config


def test_modality_ablation_one_per_modality(train_bundle, test_bundle, y_train, y_test) -> None:
    model = _model(train_bundle, y_train)
    report = ModalityAblationExplainer(MacroF1Metric()).explain(model, test_bundle, y_test)
    modalities = {c.modality for c in report.modality_contributions}
    assert modalities == set(test_bundle.modalities)
    for contribution in report.modality_contributions:
        assert contribution.score_drop == (contribution.baseline_score - contribution.ablated_score)


def test_counterfactual_changes_proba(train_bundle, test_bundle, y_train, y_test) -> None:
    model = _model(train_bundle, y_train)
    report = CounterfactualExplainer(top_k=3, sample_limit=5).explain(model, test_bundle, y_test)
    assert len(report.counterfactuals) == 5
    cf = report.counterfactuals[0]
    assert cf.original_proba.shape == cf.modified_proba.shape
    assert len(cf.removed) <= 3
