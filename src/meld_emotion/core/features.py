"""특징 컨테이너 (불변 dataclass).

특징 추출기의 출력(:class:`FeatureMatrix`)과, 한 분할(split) 전체에 대한 멀티모달 특징
묶음(:class:`FeatureBundle`)을 정의한다. 융합 분류기/평가/설명 단계는 모두 이 두 타입만을
주고받으므로 서로의 내부 구현에 의존하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from meld_emotion.core.types import (
    MODALITY_ORDER,
    UID,
    BoolArray,
    FeatureKind,
    FloatArray,
    Modality,
)


@dataclass(frozen=True, eq=False)
class FeatureMatrix:
    """한 추출기가 한 분할에 대해 산출한 (n_samples, n_features) 특징 행렬.

    ``names`` 의 길이는 열 개수와 일치해야 하며, 이는 해석(설명) 단계에서 각 열을 사람이
    읽을 수 있는 특징 이름으로 되돌리기 위함이다.
    """

    values: FloatArray
    names: tuple[str, ...]
    modality: Modality
    kind: FeatureKind
    source: str = ""

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                f"FeatureMatrix.values 는 2차원이어야 합니다 (got ndim={self.values.ndim})"
            )
        if self.values.shape[1] != len(self.names):
            raise ValueError(
                "열 개수와 names 길이가 일치하지 않습니다: "
                f"{self.values.shape[1]} != {len(self.names)}"
            )

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def select(self, rows: Sequence[int] | BoolArray) -> FeatureMatrix:
        """행(샘플) 부분집합을 선택한 새 행렬을 반환한다."""

        return FeatureMatrix(
            values=np.asarray(self.values[rows], dtype=np.float64),
            names=self.names,
            modality=self.modality,
            kind=self.kind,
            source=self.source,
        )


@dataclass(frozen=True)
class ColumnSpec:
    """스택된 설계 행렬에서 한 열의 출처(모달리티/종류/이름)."""

    modality: Modality
    kind: FeatureKind
    name: str
    source: str = ""


@dataclass(frozen=True)
class UtteranceSpec:
    """특징 행 하나가 원래 어떤 dialogue utterance 였는지에 대한 메타데이터."""

    uid: UID
    dialogue_id: int
    utterance_id: int
    speaker: str


@dataclass(frozen=True, eq=False)
class StackedFeatures:
    """여러 :class:`FeatureMatrix` 를 열 방향으로 결합한 설계 행렬.

    Early fusion 의 입력, 개념 벡터 c=[c_T,c_A,c_V], permutation 중요도 분석 등에 쓰인다.
    각 열의 출처를 ``columns`` 로 보존하여 설명 가능성을 지원한다.
    """

    values: FloatArray
    columns: tuple[ColumnSpec, ...]

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


def _empty_stack(n_samples: int) -> StackedFeatures:
    return StackedFeatures(values=np.zeros((n_samples, 0), dtype=np.float64), columns=())


@dataclass(frozen=True, eq=False)
class FeatureBundle:
    """한 분할 전체에 대한 멀티모달 특징 묶음.

    ``matrices`` 는 추출기별 출력 모음이고, ``availability`` 는 모달리티별로 각 샘플에서
    해당 모달리티가 존재하는지를 나타내는 (n_samples,) 불리언 배열이다(강건성/late fusion 용).
    """

    uids: tuple[UID, ...]
    matrices: tuple[FeatureMatrix, ...]
    availability: Mapping[Modality, BoolArray] = field(default_factory=dict)
    utterances: tuple[UtteranceSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.utterances and len(self.utterances) != len(self.uids):
            raise ValueError(
                "utterances 길이는 uids 길이와 일치해야 합니다: "
                f"{len(self.utterances)} != {len(self.uids)}"
            )

    @property
    def n_samples(self) -> int:
        return len(self.uids)

    @property
    def modalities(self) -> tuple[Modality, ...]:
        seen = {m.modality for m in self.matrices}
        return tuple(m for m in MODALITY_ORDER if m in seen)

    def by_modality(self, modality: Modality) -> tuple[FeatureMatrix, ...]:
        return tuple(m for m in self.matrices if m.modality == modality)

    def select(self, rows: Sequence[int] | BoolArray) -> FeatureBundle:
        """행(샘플) 부분집합을 선택한 새 묶음을 반환한다(설명 단계에서 사용)."""

        idx = np.asarray(rows)
        uids = tuple(str(u) for u in np.asarray(self.uids, dtype=object)[idx].tolist())
        matrices = tuple(m.select(rows) for m in self.matrices)
        availability = {mod: avail[idx] for mod, avail in self.availability.items()}
        utterances = tuple(
            np.asarray(self.utterances, dtype=object)[idx].tolist()
        ) if self.utterances else ()
        return FeatureBundle(
            uids=uids,
            matrices=matrices,
            availability=availability,
            utterances=utterances,
        )

    def by_kind(self, kind: FeatureKind) -> tuple[FeatureMatrix, ...]:
        return tuple(m for m in self.matrices if m.kind == kind)

    def stack(
        self,
        *,
        kind: FeatureKind | None = None,
        modalities: Sequence[Modality] | None = None,
    ) -> StackedFeatures:
        """선택한 특징들을 모달리티 표준 순서로 결합한 설계 행렬을 반환한다.

        결합 순서는 ``MODALITY_ORDER`` (text, audio, video) 우선, 그 다음 ``matrices`` 의
        원래 순서를 따른다 → fit/transform 간 열 순서가 항상 동일하게 유지된다.
        """

        chosen = [
            (idx, m)
            for idx, m in enumerate(self.matrices)
            if (kind is None or m.kind == kind) and (modalities is None or m.modality in modalities)
        ]
        chosen.sort(key=lambda im: (MODALITY_ORDER.index(im[1].modality), im[0]))

        if not chosen:
            return _empty_stack(self.n_samples)

        values = np.concatenate([m.values for _, m in chosen], axis=1)
        columns = tuple(
            ColumnSpec(modality=m.modality, kind=m.kind, name=name, source=m.source)
            for _, m in chosen
            for name in m.names
        )
        return StackedFeatures(values=np.asarray(values, dtype=np.float64), columns=columns)

    def embedding_matrix(self, modalities: Sequence[Modality] | None = None) -> StackedFeatures:
        """예측용 임베딩 특징만 결합 (Early fusion 입력)."""

        return self.stack(kind=FeatureKind.EMBEDDING, modalities=modalities)

    def concept_vector(self) -> StackedFeatures:
        """해석용 개념 특징만 결합 → 제안서의 c=[c_T, c_A, c_V]."""

        return self.stack(kind=FeatureKind.CONCEPT)
