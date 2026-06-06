"""MELD 데이터셋 소스."""

from __future__ import annotations

import csv
import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.status import real
from meld_emotion.core.types import Emotion, Sentiment, Split

_METADATA_SPLITS = {Split.TRAIN: "train", Split.DEV: "val", Split.TEST: "test"}


@real
class MeldDatasetSource:
    """MELD CSV 또는 baseline metadata pickle 을 `RawSample` 로 변환한다."""

    def __init__(
        self,
        root: str = "data/MELD",
        csv_train: str = "train_sent_emo.csv",
        csv_dev: str = "dev_sent_emo.csv",
        csv_test: str = "test_sent_emo.csv",
        audio_subdir: str = "audio",
        video_subdir: str = "video",
        audio_subdir_train: str | None = None,
        audio_subdir_dev: str | None = None,
        audio_subdir_test: str | None = None,
        video_subdir_train: str | None = None,
        video_subdir_dev: str | None = None,
        video_subdir_test: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        self._root = Path(root)
        self._csv = {Split.TRAIN: csv_train, Split.DEV: csv_dev, Split.TEST: csv_test}
        self._audio_subdir = audio_subdir
        self._video_subdir = video_subdir
        self._audio_subdirs = {
            Split.TRAIN: audio_subdir_train,
            Split.DEV: audio_subdir_dev,
            Split.TEST: audio_subdir_test,
        }
        self._video_subdirs = {
            Split.TRAIN: video_subdir_train,
            Split.DEV: video_subdir_dev,
            Split.TEST: video_subdir_test,
        }
        self._metadata_path = Path(metadata_path) if metadata_path is not None else None
        self._metadata: tuple[Mapping[str, object], ...] | None = None

    def load(self, split: Split) -> Iterable[RawSample]:
        split = Split(split)
        if self._metadata_path is not None:
            yield from self._load_metadata(split)
            return
        yield from self._load_csv(split)

    def _load_metadata(self, split: Split) -> Iterable[RawSample]:
        split_name = _METADATA_SPLITS[split]
        for row in self._metadata_rows():
            if str(row["split"]) != split_name:
                continue
            dialogue_id = int(str(row["dialog"]))
            utterance_id = int(str(row["utterance"]))
            yield RawSample(
                uid=_uid(split, dialogue_id, utterance_id),
                dialogue_id=dialogue_id,
                utterance_id=utterance_id,
                text=str(row["text"]),
                speaker="",
                split=split,
                mask=ModalityMask.full(),
                audio=AudioInput(sample_rate=16000),
                video=VideoInput(fps=25.0),
                emotion=Emotion(str(row["y"])),
                sentiment=None,
                metadata={
                    "source": "meld_metadata",
                    "num_words": str(row.get("num_words", "")),
                },
            )

    def _metadata_rows(self) -> tuple[Mapping[str, object], ...]:
        if self._metadata is None:
            assert self._metadata_path is not None
            with self._metadata_path.open("rb") as f:
                loaded = pickle.load(f, encoding="latin1")
            if not isinstance(loaded, list) or not loaded:
                raise ValueError(f"MELD metadata pickle 형식이 올바르지 않습니다: {self._metadata_path}")
            rows = loaded[0]
            if not isinstance(rows, list):
                raise ValueError(f"MELD metadata 첫 항목은 list 여야 합니다: {self._metadata_path}")
            self._metadata = tuple(_expect_mapping(row) for row in rows)
        return self._metadata

    def _load_csv(self, split: Split) -> Iterable[RawSample]:
        path = self._root / self._csv[split]
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogue_id = int(row["Dialogue_ID"])
                utterance_id = int(row["Utterance_ID"])
                start_time = row.get("StartTime", "")
                end_time = row.get("EndTime", "")
                yield RawSample(
                    uid=_uid(split, dialogue_id, utterance_id),
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_id,
                    text=row["Utterance"],
                    speaker=row["Speaker"],
                    split=split,
                    mask=ModalityMask.full(),
                    audio=AudioInput(
                        sample_rate=16000,
                        source_path=self._media_path(
                            split, self._audio_subdirs, self._audio_subdir, dialogue_id, utterance_id
                        ),
                        segment_start=_parse_meld_time(start_time),
                        segment_end=_parse_meld_time(end_time),
                    ),
                    video=VideoInput(
                        fps=25.0,
                        source_path=self._media_path(
                            split, self._video_subdirs, self._video_subdir, dialogue_id, utterance_id
                        ),
                    ),
                    emotion=Emotion(row["Emotion"]),
                    sentiment=Sentiment(row["Sentiment"]),
                    metadata={
                        "season": row.get("Season", ""),
                        "episode": row.get("Episode", ""),
                        "start_time": start_time,
                        "end_time": end_time,
                    },
                )

    def _media_path(
        self,
        split: Split,
        split_subdirs: Mapping[Split, str | None],
        fallback_subdir: str,
        dialogue_id: int,
        utterance_id: int,
    ) -> Path:
        subdir = split_subdirs[split] or fallback_subdir
        return self._root / subdir / _clip_name(dialogue_id, utterance_id)


def _uid(split: Split, dialogue_id: int, utterance_id: int) -> str:
    return f"{split.value}:{dialogue_id}_{utterance_id}"


def _clip_name(dialogue_id: int, utterance_id: int) -> str:
    return f"dia{dialogue_id}_utt{utterance_id}.mp4"


def _parse_meld_time(value: str) -> float | None:
    text = value.strip().strip('"')
    if not text:
        return None
    clock, _, millis = text.partition(",")
    parts = clock.split(":")
    if len(parts) != 3:
        raise ValueError(f"MELD 시간 형식이 올바르지 않습니다: {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    milliseconds = int(millis) if millis else 0
    return float(hours * 3600 + minutes * 60 + seconds) + milliseconds / 1000.0


def _expect_mapping(value: Any) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"MELD metadata row 는 mapping 이어야 합니다: {value!r}")
    return value
