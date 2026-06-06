# data — 데이터셋 소스와 레이블

`DatasetSource` 는 분할(`train`/`dev`/`test`)을 받아 `RawSample` 을 산출한다. 모델과 특징
추출기는 데이터 파일 형식을 직접 알지 않고, 이 패키지가 원천 데이터를 파이프라인 공통 타입으로
변환한다.

## 구성

- `labels.py` — `EmotionLabelEncoder`: `EMOTION_ORDER` 기준 감정 ↔ 정수 인덱스 변환.
- `synthetic.py` — 합성 데이터셋. 테스트와 예제 실험용.
- `meld.py` — MELD CSV 또는 baseline pickle metadata(`data_emotion.p`) 기반 데이터셋 소스.
- `media.py` — raw media 적재 경계. MP4 오디오는 PyAV(`av`)로 비디오 프레임을 읽지 않고 mono
  waveform 만 적재하고, MP4 비디오는 OpenCV 로 오디오를 추출하지 않고 프레임만 균등 샘플링해
  `(T,H,W,C)` 배열로 적재한다. 두 라이브러리 모두 ffmpeg 를 휠에 번들하므로 시스템 ffmpeg 없이
  in-process 로 디코딩한다.

## Raw MP4 Media Loading

`MeldDatasetSource` 는 CSV 의 `Dialogue_ID`/`Utterance_ID` 로 `dia{d}_utt{u}.mp4` 경로를
`AudioInput.source_path` 와 `VideoInput.source_path` 에 넣고, 실제 적재는 `MediaLoader` 가
담당한다. 오디오 기본값은 `audio_sample_rate=16000` 이며 mono `float64` waveform 으로
resampling 한다. CSV 의 `StartTime`/`EndTime` 은 초 단위로 파싱되어 `AudioInput` 에 보존되고,
오디오 lazy-load 후 해당 구간만 선택한다. 이미 발화/구간 단위로 잘린 MELD.Raw MP4 처럼
파일 길이가 CSV 구간 길이와 거의 같으면 전체 파일을 해당 구간으로 간주한다. 비디오 기본값은
`video_max_frames=32`, `video_frame_size=(64, 64)` 이고, 크기 의미는 `(height, width)` 다.
프레임은 BGR→RGB, `[0,1]` 범위 `float64` 로 변환된다.
`media.max_audio_seconds` 를 지정하면 실제 MP4/container 길이가 그 값을 넘는 오디오 파일은
버퍼 용량 부족을 피하기 위해 로딩 실패로 처리된다. `media.on_error: drop_sample` 과 함께 쓰면
해당 발화의 text/audio/video 특징이 모두 학습·평가에서 제외된다.

MELD.Raw 처럼 split 마다 MP4 폴더가 다르면 `audio_subdir_train`/`audio_subdir_dev`/
`audio_subdir_test` 와 `video_subdir_train`/`video_subdir_dev`/`video_subdir_test` 를 지정한다.
생략하면 기존 `audio_subdir`/`video_subdir` 값으로 fallback 한다.

```yaml
dataset:
  type: meld
  root: MELD.Raw
  csv_train: train/train_sent_emo.csv
  csv_dev: dev_sent_emo.csv
  csv_test: test_sent_emo.csv
  audio_subdir_train: train/train_splits
  audio_subdir_dev: dev_splits_complete
  audio_subdir_test: output_repeated_splits_test
```

이 lazy-load 는 해당 모달리티 추출기가 포함된 `FeaturePipeline` 에서만 실행된다. 텍스트만 쓰는
실험은 av/OpenCV 를 import 하지 않고, 오디오만 쓰는 실험은 비디오 프레임을 읽지 않으며,
비디오만 쓰는 실험은 오디오 waveform 을 읽지 않는다.

실험 설정에서는 다음처럼 조정할 수 있다:

```yaml
media:
  audio_sample_rate: 16000
  video_max_frames: 32
  video_frame_size: [64, 64]
  on_error: raise        # raise | drop_modality | drop_sample
  max_audio_seconds: 60.0
```

`media.on_error` 는 raw 파일이 없거나 깨져서 읽을 수 없는 경우의 처리 방식이다. 기본값
`raise` 는 오류를 즉시 드러낸다. `drop_modality` 는 해당 모달리티만 unavailable 로 바꾸고,
`drop_sample` 은 해당 발화 샘플 전체를 학습/평가에서 제외한다.
이 정책은 raw media lazy-load 중 발생한 `FileNotFoundError`/`ValueError` 에 적용된다.
`drop_sample` 을 쓰면 `ExperimentResult.metadata` 의 `n_train_raw`/`n_test_raw` 와
`n_train`/`n_test` 가 달라질 수 있다.

EmbeddingGemma 텍스트 임베딩과 Wav2Vec2 XLS-R 오디오 임베딩으로 MELD.Raw train/test 를 비교하는
현재 raw suite 는 split별 MP4 폴더를 위 방식으로 지정하고, train split 의 손상된 MP4 1개를
`media.on_error: drop_sample` 로 제외한다.

## Precomputed Feature Baseline

MELD 팀이 제공한 feature pickle 은 `features/precomputed.py` 의
`MeldPrecomputedFeatureExtractor` 가 읽는다. 데이터셋은 `MeldDatasetSource(metadata_path=...)` 로
`data_emotion.p` 의 발화 metadata 를 읽고, extractor 는 각 `RawSample` 의
`Dialogue_ID`/`Utterance_ID` 에 맞는 feature row 를 반환한다.
