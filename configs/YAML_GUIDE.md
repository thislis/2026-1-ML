# MELD Emotion YAML 작성 가이드

이 폴더는 실행에 바로 쓰는 대표 설정만 남긴다. 새 실험을 만들 때는 아래 파일 중 가장 가까운
예시를 복사해 수정한다.

## 남겨 둔 YAML

| 파일 | 용도 | 실행 |
| --- | --- | --- |
| `default.yaml` | 가장 가벼운 기본 smoke run. synthetic + two-stage + early centroid | `uv run meld-emotion run --config configs/default.yaml` |
| `example_synthetic.yaml` | 단일 학습 실행 예시. 외부 데이터/무거운 의존성 없이 전체 파이프라인 확인 | `uv run meld-emotion run --config configs/example_synthetic.yaml` |
| `example_suite.yaml` | 여러 실험을 한 번에 실행/비교하는 suite 예시 | `uv run meld-emotion compare --config configs/example_suite.yaml` |
| `meld_sequence_dialogue_rnn.yaml` | MELD.Raw text/audio/video sequence feature + dialogue RNN 학습 | `uv run meld-emotion run --config configs/meld_sequence_dialogue_rnn.yaml` |
| `meld_jina_omni_dialogue_rnn.yaml` | MELD.Raw Jina Omni fused multimodal embedding + dialogue RNN 학습 | `uv run meld-emotion run --config configs/meld_jina_omni_dialogue_rnn.yaml` |
| `meld_embeddinggemma_wav2vec2_suite.yaml` | MELD.Raw foundation text/audio embedding 모델 비교 suite | `uv run meld-emotion compare --config configs/meld_embeddinggemma_wav2vec2_suite.yaml` |
| `example_finegrained_xai.yaml` | sequence feature 기반 fine-grained XAI 예시 | `uv run meld-emotion run --config configs/example_finegrained_xai.yaml` |

## YAML 종류

이 프로젝트에는 두 가지 설정 형식이 있다.

### 단일 실험: `meld-emotion run`

단일 실험 YAML은 `ExperimentConfig` 하나를 표현한다.

```yaml
name: my_experiment
seed: 0
output_dir: outputs
train_split: train
eval_split: test

dataset:
  type: synthetic

extractors:
  - type: text_concepts

model:
  type: early
  base:
    type: centroid

evaluation:
  metrics: [accuracy, macro_f1, weighted_f1]
  scenarios: [full]

reporters:
  - type: console
```

실행:

```bash
uv run meld-emotion run --config configs/my_experiment.yaml
```

### 여러 실험 비교: `meld-emotion compare`

suite YAML은 `SuiteConfig`를 표현한다. `base`가 공통 설정이고, `experiments`의 각 항목이
차이만 덮어쓴다. 중첩 dict는 깊게 병합되지만, 리스트(`extractors`, `reporters` 등)는 통째로
교체된다.

```yaml
name: my_suite
metrics: [accuracy, macro_f1, weighted_f1]
robustness_metric: macro_f1
output_path: outputs/comparison.json

base:
  dataset:
    type: synthetic
  extractors:
    - { type: text_concepts }
    - { type: text_bow, n_features: 64 }
  evaluation:
    metrics: [accuracy, macro_f1, weighted_f1]
    scenarios: [full, no_text]
  reporters: []

experiments:
  - name: early_centroid
    model:
      type: early
      base: { type: centroid }

  - name: late_centroid
    model:
      type: late
      base: { type: centroid }
      combiner: { type: mean }
```

실행:

```bash
uv run meld-emotion compare --config configs/my_suite.yaml
```

## 단일 실험 최상위 필드

| 필드 | 설명 | 기본값/메모 |
| --- | --- | --- |
| `name` | 실험 이름. 결과 metadata와 로그에 기록 | `experiment` |
| `seed` | 실험 시드 메모. 모든 컴포넌트에 자동 전파되지는 않음 | `0` |
| `output_dir` | 출력 디렉터리 관례값. reporter path에 자동 적용되지는 않음 | `outputs` |
| `train_split` | 학습 split | `train` |
| `eval_split` | 평가/설명 split | `test` |
| `dataset` | 데이터셋 설정 | `synthetic` |
| `extractors` | 특징 추출기 목록 | 기본 `text_concepts` |
| `model` | 분류 모델 또는 wrapper | 기본 `early` |
| `dropout` | 학습 시 modality dropout. 없으면 미사용 | `null` |
| `media` | raw MP4 lazy-load 옵션 | 기본 16kHz, 32프레임 |
| `evaluation` | 지표, 혼동행렬, 강건성 시나리오 | 기본 4개 지표 + `full` |
| `explainers` | 설명기 목록 | 빈 목록 |
| `cache` | feature cache | `memory` |
| `reporters` | 결과 출력 방식 | `console` |

## 데이터셋

### Synthetic

외부 파일 없이 빠르게 실행되는 합성 데이터다.

```yaml
dataset:
  type: synthetic
  n_train: 350
  n_dev: 70
  n_test: 140
  seed: 0
  with_audio: true
  with_video: true
  missing_rate: 0.0
```

### MELD.Raw

MELD CSV와 MP4 파일을 쓰는 설정이다. split별 MP4 폴더가 다르면 split별 subdir를 지정한다.

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
```

precomputed MELD feature를 쓸 때는 `metadata_path`를 함께 지정한다.

```yaml
dataset:
  type: meld
  root: MELD.Features.Models
  metadata_path: MELD.Features.Models/features/data_emotion.p
```

## Raw Media 옵션

`media`는 raw MP4 오디오/비디오를 lazy-load할 때만 의미가 있다.

```yaml
media:
  audio_sample_rate: 16000
  video_max_frames: 32
  video_frame_size: [64, 64]   # [height, width]
  on_error: drop_sample        # raise | drop_modality | drop_sample
  max_audio_seconds: 60.0
  min_audio_seconds: 0.025
```

- `raise`: 파일 누락/손상/너무 짧거나 긴 오디오를 즉시 오류로 드러낸다.
- `drop_modality`: 해당 모달리티만 unavailable로 처리한다.
- `drop_sample`: 해당 발화 전체를 학습/평가에서 제외한다.

## Feature Extractor

`extractors`는 여러 개를 순서대로 나열한다. 각 extractor는 `type`으로 선택한다.

### 가벼운 synthetic/smoke용

```yaml
extractors:
  - type: text_concepts
  - type: text_bow
    n_features: 64
  - type: audio_concepts
  - type: video_concepts
```

### Foundation embedding

```yaml
extractors:
  - type: text_embeddinggemma
    model_name: google/embeddinggemma-300m
    output_dim: 768
    batch_size: 32
    normalize: true
    prompt_name: Classification
    device: null

  - type: audio_wav2vec2_xlsr
    model_name: facebook/wav2vec2-xls-r-300m
    output_dim: 1024
    batch_size: 4
    sampling_rate: 16000
    chunk_seconds: 30.0
    normalize: true
    device: null

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

필요 extra:

```bash
uv sync --extra text --extra audio --extra video
```

EmbeddingGemma는 Hugging Face gated model이므로 모델 페이지에서 라이선스 동의 후 로그인 또는
`HF_TOKEN` 설정이 필요하다.

### Sequence feature

`dialogue_rnn`과 fine-grained XAI에서 token/span/frame 단위 입력을 쓰려면 sequence extractor를
사용한다. 기존 early/late fusion은 같은 extractor의 pooled matrix를 사용한다.

```yaml
extractors:
  - type: text_token_embeddings
    model_name: bert-base-uncased
    max_tokens: 64
    output_dim: 768
    batch_size: 16
    normalize: true
    device: null

  - type: audio_wav2vec2_xlsr_sequence
    model_name: facebook/wav2vec2-xls-r-300m
    output_dim: 1024
    batch_size: 4
    sampling_rate: 16000
    max_steps: 128
    normalize: true
    device: null

  - type: video_frame_embeddings
    model_name: openai/clip-vit-base-patch32
    output_dim: 768
    batch_size: 8
    num_frames: 16
    frame_size: 224
    normalize: true
    device: null
```

### MELD precomputed feature

```yaml
extractors:
  - type: meld_precomputed
    path: MELD.Features.Models/features/text_emotion.pkl
    modality: text
    kind: embedding
    name_prefix: text_precomputed
```

`modality`는 `text`, `audio`, `video` 중 하나다. `kind`는 보통 `embedding` 또는 `concept`다.

## Model

### Early fusion

모든 feature를 하나의 행렬로 쌓아 하나의 estimator를 학습한다.

```yaml
model:
  type: early
  base:
    type: centroid
    temperature: 1.0
  use_concepts: true
```

### Late fusion

모달리티별 estimator를 학습한 뒤 확률을 결합한다.

```yaml
model:
  type: late
  base:
    type: logreg
    C: 1.0
    max_iter: 1000
  combiner:
    type: weighted
    weights:
      text: 1.0
      audio: 0.7
      video: 0.7
```

결합기:

- `mean`
- `weighted`
- `stacking` 현재 placeholder

### Two-stage wrapper

Neutral/non-neutral 판단을 명시하는 wrapper다. `base`에는 다른 classifier가 들어간다.

```yaml
model:
  type: two_stage
  neutral_threshold: 0.5
  neutral_label: neutral
  base:
    type: early
    base: { type: centroid }
```

### SVM margin two-stage

Stage 1 SVM margin이 충분히 크면 SVM 예측을 쓰고, 애매한 샘플은 Stage 2 classifier로 넘긴다.

```yaml
model:
  type: two_stage_svm_margin
  stage1:
    type: svm
    C: 1.0
    kernel: rbf
  stage2:
    type: dialogue_rnn
  margin_threshold: 0.25
  stage1_confidence_threshold: null
  stage1_use_concepts: true
  neutral_label: neutral
```

### Dialogue RNN

발화별 feature를 dialogue batch로 재구성해 PyTorch 모델을 학습한다.

```yaml
model:
  type: dialogue_rnn
  rnn_type: gru
  modality_encoder:
    text_input_dim: 0
    audio_input_dim: 0
    video_input_dim: 0
    encoder_type: conformer   # rnn | conformer
    sequence_fallback_policy: pooled
    proj_dim: 128
    hidden_dim: 128
    num_layers: 1
    num_heads: 4
    conv_kernel_size: 15
    ffn_multiplier: 4.0
    dropout: 0.2
    attention_dropout: 0.1
    pooling_type: attentive
  fusion:
    modality_dim: 128
    fusion_dim: 256
    dropout: 0.3
    use_gated_fusion: true
    use_interaction_features: true
    use_interaction: true
    gate_entropy_weight: 0.0
  dialogue_context:
    use_context: true
    use_speaker: true
    speaker_emb_dim: 32
    hidden_dim: 256
    num_layers: 1
    dropout: 0.3
  memory_attention:
    use_memory: true
    enabled: true
    hidden_dim: 256
    attn_dim: 256
    use_rope: false
    use_relative_distance_bias: true
    use_same_speaker_bias: true
    max_relative_distance: 32
  classifier:
    classifier_head_type: concat
    use_context: true
    use_memory: true
    gate_hidden_dim: 128
    gate_dropout: 0.1
    hidden_dim: 256
    dropout: 0.3
  training:
    lr: 0.0002
    weight_decay: 0.01
    gradient_clip_norm: 1.0
    batch_size: 8
    max_epochs: 50
    early_stopping_patience: 8
    validation_fraction: 0.1
    modality_dropout: 0.2
    text_dropout: 0.0
    context_dropout: 0.0
    seed: 0
    device: cpu
    best_checkpoint_path: outputs/best_model.pt
```

`*_input_dim: 0`이면 train bundle의 실제 feature 차원을 자동 추론한다. 명시할 경우 extractor
출력 차원과 정확히 맞아야 한다.

### Ensemble / MoE

```yaml
model:
  type: ensemble
  base:
    type: dialogue_rnn
  ensemble:
    mode: late_logits
    alpha: 1.0
    beta: 0.5
    gamma: 0.5
    svm_logits_path: null
    logreg_logits_path: null
    artifact_format: auto
```

```yaml
model:
  type: moe
  moe:
    enabled: true
    routing: top2
    top_k: 2
    load_balancing_loss_weight: 0.01
    expert_dropout: 0.1
    class_aware_routing: true
```

## Estimator

Early/late fusion의 `base` 또는 일부 wrapper의 stage에서 사용한다.

| type | 주요 필드 | 메모 |
| --- | --- | --- |
| `majority` | 없음 | 최빈 클래스 baseline |
| `random` | `seed` | 무작위 baseline |
| `centroid` | `temperature` | numpy 기반 경량 baseline |
| `linear_regression` | `alpha`, `fit_intercept` | one-vs-rest 선형회귀 |
| `svm` | `C`, `kernel` | sklearn 필요 |
| `logreg` | `C`, `max_iter` | sklearn 필요 |
| `random_forest` | `n_estimators`, `max_depth` | sklearn 필요 |
| `knn` | `n_neighbors` | sklearn 필요 |
| `xgboost` | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `seed` | `--extra xgboost` 필요 |
| `mlp` | `hidden_dim`, `dropout`, `learning_rate`, `batch_size`, `max_epochs`, `device` 등 | `--extra deep` 필요 |

## 학습 시 Modality Dropout

평가 시나리오가 아니라 학습 데이터 augmentation이다.

```yaml
dropout:
  drop_prob: 0.3
  seed: 0
```

## Evaluation

```yaml
evaluation:
  metrics: [accuracy, macro_f1, weighted_f1, per_class_recall]
  confusion: true
  scenarios: [full, no_text, no_audio, no_video]
```

등록된 주요 metric:

- `accuracy`
- `macro_f1`
- `weighted_f1`
- `per_class_recall`
- `nll`
- `brier_score`
- `expected_calibration_error`
- `classwise_ece`
- `confidence_bucket_accuracy`
- `high_confidence_wrong`

기본 scenario:

- `full`
- `text_only`
- `audio_only`
- `video_only`
- `no_text`
- `no_audio`
- `no_video`

새 metric은 `evaluation/metrics.py`의 `METRIC_REGISTRY`에 등록하면 YAML에서 바로 쓸 수 있다.
새 scenario는 `fusion/masking.py`의 `SCENARIOS`에 추가하면 된다.

## Explainer

```yaml
explainers:
  - type: modality_ablation
    metric: macro_f1

  - type: permutation
    metric: macro_f1
    n_repeats: 5
    seed: 0
    top_k: 20
    kinds: [concept]

  - type: counterfactual
    top_k: 5
    sample_limit: 20
```

Fine-grained XAI:

```yaml
explainers:
  - type: dialogue_finegrained_xai
    method: integrated_gradients
    n_steps: 32
    top_k: 10
    max_targets: 32
    target: predicted
```

Fine-grained XAI는 `dialogue_rnn`과 sequence extractor 조합에서 의미가 가장 크다.

## Cache

```yaml
cache:
  type: memory
```

선택지:

- `memory`: 실행 중 메모리 캐시
- `null`: 캐시 미사용
- `disk`: 현재는 실행 간 영속 캐시가 아니라 in-memory 위임 placeholder

```yaml
cache:
  type: disk
  path: .feature_cache
```

## Reporter

```yaml
reporters:
  - type: console
  - type: json
    path: outputs/result.json
  - type: dashboard
    path: outputs/dashboard.json
```

`output_dir`는 reporter path에 자동으로 붙지 않는다. 저장 위치는 각 reporter의 `path`에 직접
쓴다.

## Suite 작성 규칙

suite 최상위 필드:

```yaml
name: suite_name
metrics: [accuracy, macro_f1, weighted_f1]
robustness_metric: macro_f1
output_path: outputs/comparison.json
base: {}
experiments: []
```

주의:

- `base`와 각 experiment는 깊게 병합된다.
- `type`이 바뀌는 dict는 이전 type 전용 필드를 끌고 오지 않도록 통째로 교체된다.
- 리스트는 병합되지 않고 통째로 교체된다.
- suite runner는 같은 dataset/extractor/media/train/eval signature를 가진 실험끼리 feature cache를 공유한다.
- 어떤 변형 하나가 실패해도 suite 전체는 계속 실행하고 JSON에 실패 사유를 남긴다.

## Optional Dependency 기준

| 기능 | 설치 |
| --- | --- |
| sklearn 계열 estimator | `uv sync --extra text` 또는 관련 extra 포함 환경 |
| EmbeddingGemma text | `uv sync --extra text` |
| Wav2Vec2 audio | `uv sync --extra audio` |
| TimeSformer/VideoPrism/CLIP frame video | `uv sync --extra video` |
| Dialogue RNN / MLP | `uv sync --extra deep` |
| Fine-grained XAI / Captum | `uv sync --extra xai` |
| XGBoost | `uv sync --extra xgboost` |

일반 raw MELD sequence + XAI 조합:

```bash
uv sync --extra text --extra audio --extra video --extra deep --extra xai
```

## 작성 체크리스트

1. 단일 실행이면 `run`, 여러 실험 비교면 `compare` 형식을 고른다.
2. `dataset` split 경로가 실제 파일 구조와 맞는지 확인한다.
3. `extractors`의 출력 차원과 `dialogue_rnn.modality_encoder.*_input_dim`이 충돌하지 않게 한다.
4. 무거운 extractor를 쓰면 필요한 `uv sync --extra ...`와 Hugging Face 인증을 준비한다.
5. raw media가 깨질 가능성이 있으면 `media.on_error: drop_sample`을 고려한다.
6. 결과 파일 위치는 `reporters[].path` 또는 suite의 `output_path`에 직접 쓴다.
7. 먼저 synthetic 예시나 작은 split으로 smoke run을 통과시킨 뒤 raw MELD 전체 실험을 돌린다.

## 검증 명령

```bash
uv run meld-emotion status
uv run meld-emotion run --config configs/default.yaml
uv run meld-emotion compare --config configs/example_suite.yaml
uv run python -m pytest -q
uv run mypy src
uv run ruff check .
```

XGBoost native 테스트는 macOS arm64 OpenMP 충돌을 피하기 위해 별도 프로세스로 실행한다.

```bash
uv run python -m pytest -q -m xgboost_native
```
