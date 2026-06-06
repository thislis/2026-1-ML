"""원천 입력 데이터 컨테이너 (불변 dataclass).

파이프라인의 가장 앞단에서 데이터셋이 산출하는 단위인 :class:`RawSample` 과 그 구성요소를
정의한다. 미디어(오디오/비디오)는 경로만 들고 있다가 필요 시 적재(lazy)할 수 있도록
설계되어, 12GB 규모의 MELD 에서도 메모리 부담을 줄인다.

numpy 배열을 담는 dataclass 는 ``eq=False`` 로 두어, 동등성 비교 시 배열의 모호한 진리값
오류가 발생하지 않게 한다(동일성은 식별자 ``uid`` 로 판단).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from meld_emotion.core.types import (
    MODALITY_ORDER,
    UID,
    Emotion,
    FloatArray,
    Modality,
    Sentiment,
    Split,
)


@dataclass(frozen=True, eq=False)
class AudioInput:
    """오디오 모달리티 입력.

    ``waveform`` 이 ``None`` 이면 아직 적재되지 않은 상태이며 ``source_path`` 로 적재한다.
    ``segment_start``/``segment_end`` 는 원본 media 안에서 사용할 초 단위 구간이다.
    """

    sample_rate: int
    waveform: FloatArray | None = None
    source_path: Path | None = None
    segment_start: float | None = None
    segment_end: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self.waveform is not None


@dataclass(frozen=True, eq=False)
class VideoInput:
    """비디오 모달리티 입력. ``frames`` 형상은 (T, H, W, C)."""

    fps: float
    frames: FloatArray | None = None
    source_path: Path | None = None

    @property
    def is_loaded(self) -> bool:
        return self.frames is not None


@dataclass(frozen=True)
class ModalityMask:
    """한 샘플에서 사용 가능한 모달리티 집합.

    강건성 평가(모달리티 누락)와 학습 시 modality dropout 에 사용된다. ``frozenset`` 기반
    이므로 해시 가능하고 동등성 비교가 안전하다.
    """

    available: frozenset[Modality]

    def has(self, modality: Modality) -> bool:
        return modality in self.available

    @classmethod
    def full(cls) -> ModalityMask:
        """세 모달리티가 모두 존재하는 마스크."""

        return cls(available=frozenset(MODALITY_ORDER))

    @classmethod
    def of(cls, *modalities: Modality) -> ModalityMask:
        return cls(available=frozenset(modalities))


@dataclass(frozen=True, eq=False)
class RawSample:
    """발화(utterance) 하나에 대한 모든 원천 입력과 레이블.

    파이프라인 전반에서 데이터셋이 산출하고 특징 추출기가 소비하는 원자 단위이다.
    레이블(``emotion``/``sentiment``)은 추론 시 ``None`` 일 수 있다.
    """

    uid: UID
    dialogue_id: int
    utterance_id: int
    text: str
    speaker: str
    split: Split
    mask: ModalityMask
    audio: AudioInput | None = None
    video: VideoInput | None = None
    emotion: Emotion | None = None
    sentiment: Sentiment | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def has(self, modality: Modality) -> bool:
        """해당 모달리티가 이 샘플에서 사용 가능한지 여부."""

        return self.mask.has(modality)
