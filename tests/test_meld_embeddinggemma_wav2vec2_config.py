"""EmbeddingGemma + Wav2Vec2 XLS-R MELD suite 설정 검증."""

from __future__ import annotations

from pathlib import Path

from meld_emotion.config.loader import load_suite
from meld_emotion.config.schema import MeldConfig
from meld_emotion.core.status import ComponentStatus, status_of
from meld_emotion.core.types import Split
from meld_emotion.data.meld import MeldDatasetSource
from meld_emotion.features.audio import Wav2Vec2XlsrAudioExtractor
from meld_emotion.features.text import EmbeddingGemmaTextExtractor
from meld_emotion.pipeline.builder import build_extractor

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs" / "meld_embeddinggemma_wav2vec2_suite.yaml"


def test_suite_uses_real_embeddinggemma_and_wav2vec2_extractors() -> None:
    suite = load_suite(_CONFIG)
    assert suite.experiments
    assert all(config.media.on_error == "drop_sample" for config in suite.experiments)

    for config in suite.experiments:
        extractors = [build_extractor(extractor_config) for extractor_config in config.extractors]
        assert [type(extractor) for extractor in extractors] == [
            EmbeddingGemmaTextExtractor,
            Wav2Vec2XlsrAudioExtractor,
        ]
        assert all(status_of(type(extractor)) == ComponentStatus.REAL for extractor in extractors)


def test_suite_meld_raw_media_paths_exist_for_train_dev_test() -> None:
    suite = load_suite(_CONFIG)
    dataset = suite.experiments[0].dataset
    assert isinstance(dataset, MeldConfig)

    source = MeldDatasetSource(
        root=dataset.root,
        csv_train=dataset.csv_train,
        csv_dev=dataset.csv_dev,
        csv_test=dataset.csv_test,
        audio_subdir=dataset.audio_subdir,
        video_subdir=dataset.video_subdir,
        audio_subdir_train=dataset.audio_subdir_train,
        audio_subdir_dev=dataset.audio_subdir_dev,
        audio_subdir_test=dataset.audio_subdir_test,
        video_subdir_train=dataset.video_subdir_train,
        video_subdir_dev=dataset.video_subdir_dev,
        video_subdir_test=dataset.video_subdir_test,
    )

    for split in (Split.TRAIN, Split.DEV, Split.TEST):
        sample = next(iter(source.load(split)))
        assert sample.audio is not None and sample.audio.source_path is not None
        assert sample.video is not None and sample.video.source_path is not None
        assert sample.audio.source_path.exists()
        assert sample.video.source_path.exists()


def test_meld_csv_start_end_times_are_attached_to_audio_input() -> None:
    source = MeldDatasetSource(
        root="MELD.Raw",
        csv_train="train/train_sent_emo.csv",
        audio_subdir_train="train/train_splits",
        video_subdir_train="train/train_splits",
    )
    sample = next(iter(source.load(Split.TRAIN)))

    assert sample.audio is not None
    assert sample.audio.segment_start == 16 * 60 + 16.059
    assert sample.audio.segment_end == 16 * 60 + 21.731
