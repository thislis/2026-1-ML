# data — 데이터셋 소스와 레이블

`DatasetSource` 는 분할(`train`/`dev`/`test`)을 받아 `RawSample` 을 산출한다. 모델과 특징
추출기는 데이터 파일 형식을 직접 알지 않고, 이 패키지가 원천 데이터를 파이프라인 공통 타입으로
변환한다.

## 구성

- `labels.py` — `EmotionLabelEncoder`: `EMOTION_ORDER` 기준 감정 ↔ 정수 인덱스 변환.
- `synthetic.py` — 합성 데이터셋. 테스트와 예제 실험용.
- `meld.py` — MELD CSV 또는 baseline pickle metadata(`data_emotion.p`) 기반 데이터셋 소스.
- `media.py` — raw media 적재 경계. MP4 오디오는 librosa 로 비디오 프레임을 읽지 않고 mono
  waveform 만 적재하고, MP4 비디오는 OpenCV 로 오디오를 추출하지 않고 프레임만 균등 샘플링해
  `(T,H,W,C)` 배열로 적재한다.

## Raw MP4 Media Loading

`MeldDatasetSource` 는 CSV 의 `Dialogue_ID`/`Utterance_ID` 로 `dia{d}_utt{u}.mp4` 경로를
`AudioInput.source_path` 와 `VideoInput.source_path` 에 넣고, 실제 적재는 `MediaLoader` 가
담당한다. 오디오 기본값은 `audio_sample_rate=16000` 이며 mono `float64` waveform 으로
resampling 한다. 비디오 기본값은 `video_max_frames=32`, `video_frame_size=(64, 64)` 이고,
크기 의미는 `(height, width)` 다. 프레임은 BGR→RGB, `[0,1]` 범위 `float64` 로 변환된다.

이 lazy-load 는 해당 모달리티 추출기가 포함된 `FeaturePipeline` 에서만 실행된다. 텍스트만 쓰는
실험은 librosa/OpenCV 를 import 하지 않고, 오디오만 쓰는 실험은 비디오 프레임을 읽지 않으며,
비디오만 쓰는 실험은 오디오 waveform 을 읽지 않는다.

실험 설정에서는 다음처럼 조정할 수 있다:

```yaml
media:
  audio_sample_rate: 16000
  video_max_frames: 32
  video_frame_size: [64, 64]
```

## Precomputed Feature Baseline

MELD 팀이 제공한 feature pickle 은 `features/precomputed.py` 의
`MeldPrecomputedFeatureExtractor` 가 읽는다. 데이터셋은 `MeldDatasetSource(metadata_path=...)` 로
`data_emotion.p` 의 발화 metadata 를 읽고, extractor 는 각 `RawSample` 의
`Dialogue_ID`/`Utterance_ID` 에 맞는 feature row 를 반환한다.
