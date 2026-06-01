"""합성 데이터셋 소스의 결정성과 구조."""

from __future__ import annotations

from meld_emotion.core.types import EMOTION_ORDER, Modality, Split
from meld_emotion.data.synthetic import SyntheticDatasetSource


def test_sizes() -> None:
    source = SyntheticDatasetSource(n_train=21, n_dev=7, n_test=14, seed=0)
    assert len(list(source.load(Split.TRAIN))) == 21
    assert len(list(source.load(Split.DEV))) == 7
    assert len(list(source.load(Split.TEST))) == 14


def test_deterministic() -> None:
    a = list(SyntheticDatasetSource(n_train=10, seed=3).load(Split.TRAIN))
    b = list(SyntheticDatasetSource(n_train=10, seed=3).load(Split.TRAIN))
    assert [s.uid for s in a] == [s.uid for s in b]
    assert [s.text for s in a] == [s.text for s in b]
    assert a[0].audio is not None and b[0].audio is not None
    assert a[0].audio.waveform is not None and b[0].audio.waveform is not None
    assert (a[0].audio.waveform == b[0].audio.waveform).all()


def test_all_emotions_present() -> None:
    samples = list(SyntheticDatasetSource(n_train=70, seed=1).load(Split.TRAIN))
    present = {s.emotion for s in samples}
    assert present == set(EMOTION_ORDER)


def test_modalities_available_by_default() -> None:
    sample = next(iter(SyntheticDatasetSource(seed=0).load(Split.TRAIN)))
    assert sample.has(Modality.TEXT)
    assert sample.has(Modality.AUDIO)
    assert sample.has(Modality.VIDEO)


def test_missing_rate_drops_modalities() -> None:
    samples = list(SyntheticDatasetSource(n_train=200, seed=2, missing_rate=0.5).load(Split.TRAIN))
    # 절반가량의 샘플에서 오디오/비디오가 빠져야 한다(텍스트는 항상 존재).
    audio_present = sum(s.has(Modality.AUDIO) for s in samples)
    assert all(s.has(Modality.TEXT) for s in samples)
    assert 0 < audio_present < len(samples)
