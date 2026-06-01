"""텍스트 개념 추출기의 구체 값 검증."""

from __future__ import annotations

from meld_emotion.core.data import ModalityMask, RawSample
from meld_emotion.core.types import FeatureKind, Modality, Split
from meld_emotion.features.text.concepts import TextConceptExtractor


def _sample(text: str) -> RawSample:
    return RawSample(
        uid="x",
        dialogue_id=0,
        utterance_id=0,
        text=text,
        speaker="s",
        split=Split.TRAIN,
        mask=ModalityMask.of(Modality.TEXT),
    )


def test_concept_names_and_kind() -> None:
    extractor = TextConceptExtractor()
    matrix = extractor.transform([_sample("hello")])
    assert matrix.kind == FeatureKind.CONCEPT
    assert matrix.modality == Modality.TEXT
    assert "negation_count" in matrix.names
    assert matrix.values.shape == (1, len(matrix.names))


def test_counts() -> None:
    extractor = TextConceptExtractor()
    matrix = extractor.transform([_sample("I am not happy! not really?")])
    row = dict(zip(matrix.names, matrix.values[0], strict=True))
    assert row["exclamation_count"] == 1.0
    assert row["question_count"] == 1.0
    assert row["negation_count"] == 2.0  # 두 번의 "not"
    assert row["positive_ratio"] > 0.0  # "happy"
