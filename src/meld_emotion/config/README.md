# config — 실험 설정

타입이 명시된 **설정 dataclass** 가 단일 진실 공급원이다. 파이썬에서 직접 생성하면 mypy/IDE
검사를 온전히 받고, 동일한 구조를 YAML 로 읽고 쓸 수도 있다(하이브리드 방식).

## 한 실험을 기술하는 것: `ExperimentConfig`

한 실험 = 하나의 `ExperimentConfig`. 아래가 최상위 필드 전부이며, 이것이 곧 "바꿀 수 있는
실험 변수" 목록이다. 절차(단계 순서) 자체는 설정이 아니라 [runner.py](../pipeline/runner.py)
`ExperimentRunner.run()` 에 있다.

| 필드 | 역할 | 기본값 |
| --- | --- | --- |
| `name` | 실험 이름(리포트·결과에 기록) | `"experiment"` |
| `seed` | 의도된 전역 시드. **현재는 각 컴포넌트의 자체 시드를 쓰며 자동 전파되지 않음** | `0` |
| `output_dir` | 출력 디렉터리(관례용). **리포터 경로에 자동 반영되지 않음** | `"outputs"` |
| `train_split` | 학습 분할 | `"train"` |
| `eval_split` | 평가·강건성·설명을 수행할 분할(예: `"dev"`) | `"test"` |
| `dataset` | 데이터셋 소스(`synthetic`/`meld`) | `synthetic` |
| `extractors` | 특징 추출기 목록(모달리티 × 임베딩/개념) | `(text_concepts,)` |
| `model` | 분류기(`early`/`late`, `dialogue_rnn`) | `early` |
| `dropout` | 학습 시 modality dropout(`None` = 미적용) | `None` |
| `media` | raw MP4 lazy-load(`audio_sample_rate`, `video_max_frames`, `video_frame_size`, `on_error`, audio 길이 상/하한) | 16kHz, 32프레임, 64×64, `raise`, 길이 제한 없음 |
| `evaluation` | 지표·혼동행렬·강건성 시나리오 | 기본 4지표, `full` |
| `explainers` | 설명기 목록(permutation/ablation/counterfactual) | `()` |
| `cache` | 특징 캐시(`memory`/`null`/`disk`) | `memory` |
| `reporters` | 리포터 목록(`console`/`json`/`dashboard`) | `(console,)` |

> `seed`/`output_dir` 은 현재 장식적(decorative)이다. 재현성은 `dataset.seed`,
> `dropout.seed`, `explainer.seed` 등 컴포넌트별 시드가 책임지고, 출력 경로는 각 리포터의
> `path` 가 결정한다.

## 구성

- `schema.py` — 모든 설정 dataclass 와 카테고리 레지스트리(`DATASET_CONFIGS`,
  `EXTRACTOR_CONFIGS`, `ESTIMATOR_CONFIGS`, `MODEL_CONFIGS`, `EXPLAINER_CONFIGS`, ...).
  각 설정은 다형성 식별자 `type` 을 `ClassVar` 로 가진다(생성자 인자가 아니라 YAML 경계 전용).
- `loader.py` — `load_config`/`dump_config`(파일), `from_dict`/`to_dict`(직렬화). `type` 을 읽어
  레지스트리에서 알맞은 dataclass 를 복원하며 중첩 설정(model.base, late.combiner,
  stacking.meta, dialogue_rnn 의 하위 설정)도 재귀 처리한다. 다중 실험 비교는 `SuiteConfig` +
  `load_suite`/`suite_from_dict` (공유 `base` + 변형 목록 깊은 병합) — 형식은
  [pipeline/README.md](../pipeline/README.md).

## `dialogue_rnn` 모델 설정

PyTorch dialogue 모델은 `model.type: dialogue_rnn` 으로 선택한다. `modality_encoder`,
`fusion`, `dialogue_context`, `memory_attention`, `classifier`, `training` 하위 설정을 가진다.
`modality_encoder.text_input_dim`/`audio_input_dim`/`video_input_dim` 은 기본 `0` 이며 실제
특징 차원을 자동 추론한다. RoPE 는 `memory_attention.use_rope: false` 가 기본이다.

```yaml
model:
  type: dialogue_rnn
  rnn_type: gru
  memory_attention:
    use_rope: false
    use_relative_distance_bias: true
    use_same_speaker_bias: true
  training:
    lr: 0.0002
    batch_size: 8
    max_epochs: 50
    modality_dropout: 0.2
    best_checkpoint_path: outputs/best_model.pt
```

`training.best_checkpoint_path` 를 지정하면 각 epoch 뒤 weighted F1 을 계산해 지금까지 가장 좋은
`dialogue_rnn` 모델을 해당 경로에 checkpoint 로 저장한다. validation split 이 있으면 validation
점수를, `validation_fraction: 0.0` 처럼 검증 분할이 없으면 train 점수를 기준으로 삼는다.

## 현재 등록된 주요 타입

- 데이터셋: `synthetic`, `meld`
- 특징: `text_concepts`, `text_bow`, `text_tfidf`, `text_embeddings`,
  `text_embeddinggemma`, `text_token_embeddings`, `audio_concepts`, `audio_mfcc`,
  `audio_wav2vec2_xlsr`, `audio_wav2vec2_xlsr_sequence`, `video_concepts`,
  `video_visual`, `video_timesformer`, `video_videoprism`, `video_frame_embeddings`,
  `meld_precomputed`
- 모델: `early`, `late`, `dialogue_rnn`, `ensemble`, `moe`, `two_stage`,
  `two_stage_svm_margin`
- 기초 학습기: `majority`, `random`, `centroid`, `linear_regression`, `svm`, `logreg`,
  `random_forest`, `knn`, `xgboost`, `mlp`
- 결합기: `mean`, `weighted`, `stacking`
- 설명기: `permutation`, `modality_ablation`, `counterfactual`, `dialogue_finegrained_xai`
- 캐시: `memory`, `null`, `disk`
- 리포터: `console`, `json`, `dashboard`

## 여러 실험을 기술하는 것: `SuiteConfig`

`meld-emotion compare` 는 `SuiteConfig` 를 읽어 여러 `ExperimentConfig` 를 같은 실행 경로로
돌린다. suite YAML 의 최상위 필드는 `name`, `metrics`, `robustness_metric`, `output_path`,
공유 설정 `base`, 변형 목록 `experiments` 이다. `base` 와 각 변형은 깊은 병합되지만,
`extractors` 같은 리스트는 통째로 교체되고, 다형성 `type` 이 바뀌는 매핑은 이전 타입의 전용
필드를 끌고 오지 않도록 통째로 교체된다.

## 새 설정 항목 추가하기

1. `schema.py` 에 해당 베이스를 상속한 frozen dataclass 를 만들고 `type: ClassVar[str]` 지정.
2. 같은 파일에서 카테고리 레지스트리에 등록: `XXX_CONFIGS.add(MyConfig.type, MyConfig)`.
3. 그 설정을 구체 객체로 바꾸는 연결을 [pipeline/builder.py](../pipeline/builder.py) 에 추가.

```python
@dataclass(frozen=True)
class MyExtractorConfig(ExtractorConfig):
    type: ClassVar[str] = "text_myfeat"
    dim: int = 64
EXTRACTOR_CONFIGS.add(MyExtractorConfig.type, MyExtractorConfig)
```

YAML 에서는 `{type: text_myfeat, dim: 64}` 로 사용한다.

EmbeddingGemma 텍스트 임베딩은 다음처럼 선택한다. `output_dim` 은 128/256/512/768 중 하나이며,
실행 환경에는 `uv sync --extra text` 와 Hugging Face 의 Google Gemma 라이선스 동의가 필요하다.
모델은 gated repository 이므로 [모델 페이지](https://huggingface.co/google/embeddinggemma-300m)
에서 라이선스에 동의한 뒤 Read token 으로 로그인해야 한다.

```bash
uv run huggingface-cli login
uv run huggingface-cli whoami
```

비대화형 실행에서는 `HF_TOKEN=hf_...` 환경변수를 설정한다.

```yaml
extractors:
  - type: text_embeddinggemma
    model_name: google/embeddinggemma-300m
    output_dim: 768
    batch_size: 32
    normalize: true
    prompt_name: Classification
    device: null
```

`prompt_name` 은 EmbeddingGemma 모델의 prompt dictionary key 와 대소문자까지 일치해야 한다.
기본 분류 목적 key 는 `Classification` 이다.

Wav2Vec2 XLS-R 오디오 임베딩은 다음처럼 선택한다. 실행 환경에는 `uv sync --extra audio` 가
필요하고, 입력 waveform 은 16kHz mono 여야 한다. `facebook/wav2vec2-xls-r-300m` 은 ASR
tokenizer 가 없는 base checkpoint 이므로 내부 로딩은 `AutoFeatureExtractor` + `Wav2Vec2Model`
경로를 사용한다.

```yaml
extractors:
  - type: audio_wav2vec2_xlsr
    model_name: facebook/wav2vec2-xls-r-300m
    output_dim: 1024
    batch_size: 4
    sampling_rate: 16000
    chunk_seconds: 30.0
    normalize: true
    device: null
```

TimeSformer 비디오 임베딩은 다음처럼 선택한다. 실행 환경에는 `uv sync --extra video` 가
필요하고, 내부적으로 Hugging Face Transformers 의 `TimesformerModel` 로
`facebook/timesformer-base-finetuned-k400` checkpoint 를 lazy-load 한다. MELD raw MP4 는
`MediaLoader` 가 프레임을 lazy-load 하고, extractor 가 `num_frames` 개를 균등 샘플링해
`frame_size`×`frame_size` RGB 입력으로 맞춘 뒤 ImageNet mean/std 정규화를 적용한다. 기본
pooling 은 CLS token 이며, `pooling: mean` 으로 token 평균 pooling 을 선택할 수 있다.

```yaml
extractors:
  - type: video_timesformer
    model_name: facebook/timesformer-base-finetuned-k400
    output_dim: 768
    batch_size: 2
    num_frames: 8
    frame_size: 224
    normalize: true
    pooling: cls
    device: null
```

VideoPrism 비디오 임베딩은 다음처럼 선택한다. 실행 환경에는 `uv sync --extra video` 가 필요하고,
내부적으로 Google DeepMind VideoPrism JAX/Flax 구현의
`google/videoprism-base-f16r288` checkpoint 를 사용한다. MELD raw MP4 는 `MediaLoader` 가
프레임을 lazy-load 하고, extractor 가 `num_frames` 개를 균등 샘플링해 `frame_size`×`frame_size`
RGB 입력으로 맞춘다.
macOS arm64 에서는 upstream `videoprism` 의 `tensorflow-cpu` 의존성이 wheel 부재로 설치를
깨뜨릴 수 있으므로, 이 프로젝트는 `pyproject.toml` 의 uv dependency metadata override 로
`tensorflow-cpu` 를 제외하고 extractor 안의 `tensorflow.io.gfile` shim 으로 필요한 import 만
대체한다. 따라서 `uv sync --extra all --extra dev` 경로에서도 VideoPrism 설정을 함께 둘 수 있다.

```yaml
extractors:
  - type: video_videoprism
    model_name: google/videoprism-base-f16r288
    output_dim: 768
    num_frames: 16
    frame_size: 288
    normalize: true
    prefer_batched_input: true
```

세 sequence feature 를 모두 쓰는 raw MELD dialogue 설정은
[configs/meld_sequence_dialogue_rnn.yaml](../../../configs/meld_sequence_dialogue_rnn.yaml) 이다. 이
설정은 `text_token_embeddings`, `audio_wav2vec2_xlsr_sequence`, `video_frame_embeddings` 를 함께
쓰고 `dialogue_rnn` 의 Conformer modality encoder 경로를 학습한다.

Fine-grained dialogue XAI 는 sequence extractor 와 `dialogue_finegrained_xai` 설명기를 함께 쓴다.

```yaml
extractors:
  - type: text_token_embeddings
    model_name: bert-base-uncased
    max_tokens: 64
  - type: audio_wav2vec2_xlsr_sequence
    max_steps: 128
  - type: video_frame_embeddings
    num_frames: 16
model:
  type: dialogue_rnn
explainers:
  - type: dialogue_finegrained_xai
    n_steps: 32
    top_k: 10
    max_targets: 32
reporters:
  - type: json
    path: outputs/finegrained_xai.json
  - type: dashboard
    path: outputs/finegrained_xai_dashboard.json
```

raw media 오류 처리는 `media.on_error` 로 조정한다. 기본값 `raise` 는 파일 누락/손상 시 실험을
중단하고, `drop_modality` 는 해당 샘플의 해당 모달리티만 missing 으로 처리한다. `drop_sample`
은 해당 발화 샘플 전체를 학습/평가에서 제외한다. 러너 metadata 에는 raw 로드 전 개수
(`n_train_raw`/`n_test_raw`)와 실제 특징화 후 개수(`n_train`/`n_test`)가 함께 기록된다.
`media.max_audio_seconds` 를 지정하면 실제 MP4/container 길이가 그 값을 초과하는 audio media 를
로딩 실패로 처리한다. `drop_sample` 정책과 함께 쓰면 버퍼 용량 부족을 일으키는 긴 MP4와 그에
대응되는 text 가 모두 학습·평가에서 제외된다.
`media.min_audio_seconds` 는 CSV 구간 선택 후 너무 짧은 waveform 을 같은 방식으로 제외해
Wav2Vec2 convolution kernel 오류를 피한다.

MELD.Raw 의 train/test MP4 폴더가 split 별로 다른 경우에는 `MeldConfig` 에 split별 media
subdir 를 지정한다. EmbeddingGemma 텍스트 임베딩과 Wav2Vec2 XLS-R 오디오 임베딩을 함께 쓰는
모델 비교 suite 는 [configs/meld_embeddinggemma_wav2vec2_suite.yaml](../../../configs/meld_embeddinggemma_wav2vec2_suite.yaml)
에 있다.

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
  video_subdir_train: train/train_splits
  video_subdir_dev: dev_splits_complete
  video_subdir_test: output_repeated_splits_test
media:
  on_error: drop_sample
  max_audio_seconds: 60.0
  min_audio_seconds: 0.025
```

## 주의

- `ClassVar` 식별자는 필드 순서/기본값 문제를 피하려는 의도다(중첩 기본값은 `default_factory`).
- 새 스칼라 필드는 **기본값**을 주어야 기존 YAML 과 호환된다.
- 새 중첩 설정을 YAML 에서 복원해야 한다면 `loader.py` 에 재귀 복원 함수를 추가해야 한다
  (`model.base`, `late.combiner`, `stacking.meta`, `dialogue_rnn` 의 하위 설정,
  `media.video_frame_size`/`media.on_error`/`media.max_audio_seconds`/
  `media.min_audio_seconds` 가 현재 예시다).
