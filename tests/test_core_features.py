"""FeatureMatrix / FeatureBundle 불변식과 헬퍼."""

from __future__ import annotations

import numpy as np
import pytest

from meld_emotion.core.features import (
    FeatureBundle,
    FeatureMatrix,
    FeatureUnit,
    SequenceFeatureMatrix,
)
from meld_emotion.core.types import FeatureKind, Modality


def _matrix(modality: Modality, kind: FeatureKind, n: int, k: int, base: float) -> FeatureMatrix:
    values = np.full((n, k), base, dtype=np.float64)
    names = tuple(f"{modality.value}_{i}" for i in range(k))
    return FeatureMatrix(values=values, names=names, modality=modality, kind=kind)


def test_feature_matrix_rejects_name_mismatch() -> None:
    with pytest.raises(ValueError):
        FeatureMatrix(
            values=np.zeros((3, 2)),
            names=("a",),
            modality=Modality.TEXT,
            kind=FeatureKind.CONCEPT,
        )


def test_feature_matrix_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        FeatureMatrix(
            values=np.zeros(3),
            names=(),
            modality=Modality.TEXT,
            kind=FeatureKind.CONCEPT,
        )


def test_stack_orders_by_modality() -> None:
    bundle = FeatureBundle(
        uids=("a", "b"),
        matrices=(
            _matrix(Modality.VIDEO, FeatureKind.EMBEDDING, 2, 1, 3.0),
            _matrix(Modality.TEXT, FeatureKind.EMBEDDING, 2, 1, 1.0),
            _matrix(Modality.AUDIO, FeatureKind.EMBEDDING, 2, 1, 2.0),
        ),
    )
    stacked = bundle.embedding_matrix()
    # text, audio, video 순서로 결합되어야 한다.
    assert [c.modality for c in stacked.columns] == [
        Modality.TEXT,
        Modality.AUDIO,
        Modality.VIDEO,
    ]
    assert stacked.values[0].tolist() == [1.0, 2.0, 3.0]


def test_concept_vector_filters_kind() -> None:
    bundle = FeatureBundle(
        uids=("a", "b"),
        matrices=(
            _matrix(Modality.TEXT, FeatureKind.CONCEPT, 2, 2, 1.0),
            _matrix(Modality.TEXT, FeatureKind.EMBEDDING, 2, 3, 9.0),
        ),
    )
    concept = bundle.concept_vector()
    assert concept.n_features == 2
    assert all(c.kind == FeatureKind.CONCEPT for c in concept.columns)


def test_bundle_select_rows() -> None:
    bundle = FeatureBundle(
        uids=("a", "b", "c"),
        matrices=(_matrix(Modality.TEXT, FeatureKind.CONCEPT, 3, 2, 1.0),),
        availability={Modality.TEXT: np.array([True, False, True])},
    )
    selected = bundle.select([0, 2])
    assert selected.uids == ("a", "c")
    assert selected.matrices[0].n_samples == 2
    assert selected.availability[Modality.TEXT].tolist() == [True, True]


def test_sequence_feature_matrix_validates_and_selects_rows() -> None:
    sequence = SequenceFeatureMatrix(
        values=np.ones((3, 4, 2), dtype=np.float64),
        mask=np.array(
            [[True, True, False, False], [True, False, False, False], [True, True, True, False]]
        ),
        units=(
            (FeatureUnit("a", 0), FeatureUnit("b", 1)),
            (FeatureUnit("c", 0),),
            (FeatureUnit("d", 0), FeatureUnit("e", 1), FeatureUnit("f", 2)),
        ),
        names=("x", "y"),
        modality=Modality.TEXT,
        kind=FeatureKind.EMBEDDING,
    )
    selected = sequence.select([0, 2])
    assert selected.values.shape == (2, 4, 2)
    assert selected.units[1][2].label == "f"


def test_bundle_select_preserves_sequence_matrices() -> None:
    sequence = SequenceFeatureMatrix(
        values=np.ones((2, 2, 1), dtype=np.float64),
        mask=np.ones((2, 2), dtype=bool),
        units=((FeatureUnit("a", 0), FeatureUnit("b", 1)),) * 2,
        names=("x",),
        modality=Modality.TEXT,
        kind=FeatureKind.EMBEDDING,
    )
    bundle = FeatureBundle(
        uids=("a", "b"),
        matrices=(_matrix(Modality.TEXT, FeatureKind.EMBEDDING, 2, 1, 1.0),),
        sequence_matrices=(sequence,),
    )
    selected = bundle.select([1])
    assert selected.sequence_matrices[0].n_samples == 1
    assert selected.sequence_matrices[0].units[0][0].label == "a"
