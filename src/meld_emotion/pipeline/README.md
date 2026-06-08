# pipeline — 오케스트레이션과 구성 루트

데이터→특징→융합→평가→설명→리포트를 하나로 엮는 상위 계층. 오직 `core` 의 Protocol 에만
의존한다(DIP). 구체 구현 연결은 `builder.py` 한 곳에 모인다.

## 구성

- `cache.py` — `InMemoryFeatureCache`·`NullFeatureCache` (완전 구현), `DiskFeatureCache` (임시,
  인메모리로 위임). 추출-1회·재사용-N회. `InMemoryFeatureCache` 는 개별 `FeatureMatrix` 와
  media chunk 경로의 `FeatureBundle` 캐시를 모두 지원한다.
- `feature_pipeline.py` — `FeaturePipeline`: 추출기들을 학습 분할로 `fit` 후 임의 분할을
  `FeatureBundle` 로 변환하고 모달리티 가용성 마스크를 구성한다. 오디오/비디오 추출기가 있고
  source path 만 있는 샘플은 특징 추출 전 `MediaLoader` 로 필요한 배열을 lazy-load 한다.
  각 행의 `dialogue_id`/`utterance_id`/`speaker` 는 `FeatureBundle.utterances` 에 보존된다.
- `runner.py` — `ExperimentRunner`: 한 실험을 끝까지 실행하고 `ExperimentResult` 반환.
- `builder.py` — **구성 루트**. `build_experiment(config) -> ExperimentRunner`. 유일하게 모든
  구체 구현을 import 하여 설정→객체로 연결한다.
- `suite.py` — `SuiteRunner`: 여러 `ExperimentConfig` 를 각각 `build_experiment().run()` 으로
  실행해 `ComparisonReport` 로 모은다(기존 단일 실행 경로를 재사용하는 얇은 층). dataset,
  extractors, media, train/eval split 이 같은 실험끼리는 suite 내부 in-memory feature cache 를
  공유한다.
- `meld_emotion.inference` — 학습 runner 와 별개로 단일 MP4+텍스트를 `RawSample` 하나로 감싸
  `FeaturePipeline` 과 저장된 `dialogue_rnn` checkpoint 를 재사용해 감정을 예측한다.

## 흐름

```
source.load → feature_pipeline.fit_transform(train) → classifier.fit
            → feature_pipeline.transform(test) → evaluator/robustness/explainers
            → ExperimentResult → reporters
```

오디오/비디오 특징 추출기가 포함된 MELD 실험에서는 흐름 앞단에서 MP4 의 필요한 스트림만
적재한다. 오디오 extractor 는 비디오 프레임을 읽지 않고 mono waveform 만, 비디오 extractor 는
오디오를 추출하지 않고 프레임만 받는다. 기본은 16kHz 오디오, 32프레임 균등 샘플링 + 64×64
resize 이며 `ExperimentConfig.media` 또는 YAML 의 `media.audio_sample_rate`,
`media.video_max_frames`, `media.video_frame_size` 로 바꾼다.
`video_timesformer` 는 이 프레임을 8프레임, 224×224 ImageNet 정규화 입력으로 맞춰
`facebook/timesformer-base-finetuned-k400` CLS token 또는 token 평균 embedding 을 만든다.
`video_videoprism` 은 이 프레임을 16프레임, 288×288 입력으로 재샘플/resize 한 뒤
`google/videoprism-base-f16r288` patch token 을 평균 풀링해 발화 단위 embedding 을 만든다.

## 여러 실험 비교 (suite)

`meld-emotion compare --config <suite.yaml>` 는 여러 실험을 한 번에 실행하고 지표·강건성
비교표(콘솔 + JSON)를 낸다. 출력은 `reporting.report.ComparisonReporter` 가 담당하며,
교수님 피드백의 Early/Late fusion 비교, SVM/LogReg 베이스라인 비교 등을 한 명령으로 수행하기
위한 것이다.

suite YAML 은 모든 변형이 공유하는 `base` 와, 그 위에 **차이만** 덮어쓰는 `experiments`
목록으로 구성된다(매핑은 깊게 병합, 리스트는 통째 교체). 각 변형은 고유한 `name` 을 가져야
한다(비교표의 키).

```yaml
name: fusion_comparison
metrics: [accuracy, macro_f1, weighted_f1]   # 비교표에 나열할 스칼라 지표
robustness_metric: macro_f1                  # 강건성 표에 쓸 지표
output_path: outputs/comparison.json

base:                                        # 공유 설정 (ExperimentConfig 의 부분 dict)
  dataset: { type: synthetic, n_train: 350, n_test: 140 }
  evaluation: { scenarios: [full, no_text, no_audio, no_video] }
  reporters: []                              # 변형별 콘솔 출력은 끄고 집계표만

experiments:                                 # base 위에 차이만
  - { name: early_centroid, model: { type: early, base: { type: centroid } } }
  - { name: late_centroid,  model: { type: late,  base: { type: centroid }, combiner: { type: mean } } }
```

**경계 내성**: 어떤 변형이 미구현 경계, 모델 다운로드, gated model 인증, native library 같은
외부 경계에서 예외를 던져도 비교 전체는 멈추지 않는다. 그 변형은 `[Failed]` 에 사유와 함께
기록되고 나머지는 정상 비교된다. 예제: `configs/example_suite.yaml`.

## 새 컴포넌트를 파이프라인에 연결하기

새 데이터셋/추출기/모델/결합기/설명기/캐시/리포터는 각 패키지에서 구현·설정 등록 후, 여기
`builder.py` 의 대응 `build_*` 함수에 분기를 한 줄 추가하면 끝이다. `ExperimentRunner` 와
`core` 계약은 건드리지 않는다(OCP).

## 메모

- 실험 절차(단계/순서)는 `runner.py` 의 `ExperimentRunner.run()` 한 곳에 있다. 새 절차(dev 기반
  모델 선택, 교차검증, 다중 시드 등)는 여기서 바꾼다.
- 학습/평가 분할, 학습 시 modality dropout, raw MP4 미디어 적재 옵션은 설정으로 제어한다
  (`ExperimentConfig` 의 `train_split`/`eval_split`/`dropout`/`media`). 빌더가 이를 러너에
  연결하므로 YAML 만으로 켤 수 있다:

  ```yaml
  train_split: train
  eval_split: test
  dropout:        # 생략 시 미적용
    drop_prob: 0.3
    seed: 0
  media:
    audio_sample_rate: 16000
    video_max_frames: 32
    video_frame_size: [64, 64]   # [height, width]
    on_error: drop_sample        # raise | drop_modality | drop_sample
    max_audio_seconds: 60.0      # 초과 audio media 는 로딩 실패로 처리
    min_audio_seconds: 0.025     # 너무 짧은 Wav2Vec2 입력도 제외 가능
  ```
- 특징 캐시 키는 `"{extractor.name}|{split}"` 이며, media chunk 경로에서는 split/uid fingerprint
  기반 bundle cache 도 함께 사용한다.
- `dialogue_rnn` 모델은 `FeatureBundle.utterances` 를 기준으로 발화별 특징 행을 dialogue batch
  `[B,N,1,D]` 로 재구성한다. 기존 추출기는 발화별 single vector 를 내므로 sequence length 는
  1이며, padding 발화는 loss/evaluation 에서 제외된다.
- `ExperimentRunner` metadata 는 raw 샘플 수(`n_train_raw`/`n_test_raw`)와 media error policy
  적용 후 실제 feature bundle 샘플 수(`n_train`/`n_test`)를 함께 기록한다.
- `meld-emotion run` 과 `meld-emotion compare` 는 기본 `INFO` 로그를 stderr 로 출력하며,
  `--log-level DEBUG` 와 `--log-file <path>` 로 상세 로그와 파일 로그를 켤 수 있다.
- `meld-emotion infer --mp4 <path> --text <text>` 는 `outputs/best_model.pt` 를 기본 checkpoint 로
  사용해 단일 발화 감정을 출력한다. 같은 기능은 루트의 `infer_emotion.py` wrapper 로도 실행할 수
  있다.
