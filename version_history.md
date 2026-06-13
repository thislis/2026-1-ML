# Version History

## ours_v3pre

`ours_v3pre` 는 `ours_v2.5` 의 tri-modal raw/sequence pipeline 위에 **fused multimodal
embedding** 경로를 추가한 pre-release 버전이다. 핵심 변화는
`jinaai/jina-embeddings-v5-omni-small` 을 사용하는 `jina_omni_multimodal` extractor 와,
그 단일 `Modality.MULTIMODAL` feature 를 dialogue-level 모델에 바로 넣는
`dialogue_rnn.input_mode: multimodal` 이다. 즉, v2.5 가 text/audio/video 를 각각 추출한 뒤
모델 내부에서 fusion 했다면, v3pre 는 foundation model 단계에서 텍스트·오디오·비디오를 먼저
하나의 1024차원 embedding 으로 합치고, 그 embedding sequence 를 dialogue context/memory/classifier
로 학습하는 경로를 열었다.

현재 코드 기준 구현 상태는 `uv run meld-emotion status` 로 확인한다. 이 문서를 갱신한 시점의
상태는 전체 69개 컴포넌트 중 REAL 62개, PLACEHOLDER 7개, UNIMPLEMENTED 0개다. `ours_v2.5`
대비 REAL 구현이 하나 늘었고, 새 REAL 컴포넌트는
`features.multimodal.jina_omni.JinaOmniMultimodalExtractor` 이다. 기존 placeholder 경계는
TF-IDF, sentence embedding, MFCC, visual cue, stacking combiner, disk cache, dashboard HTML
rendering 으로 유지된다.

### v2.5에서 v3pre로 넘어온 변화

- fused multimodal extractor: `jina_omni_multimodal` 을 추가해 텍스트, 오디오 waveform/path,
  비디오 frame/path 를 SentenceTransformers remote-code Jina Omni 모델에 함께 전달한다.
- multimodal feature type: `Modality.MULTIMODAL` 을 도메인 enum 과 `MODALITY_ORDER`,
  `FeatureBundle.availability` 흐름에 추가했다.
- dialogue model single-stream mode: `dialogue_rnn.input_mode: multimodal` 을 추가해
  tri-modal encoder/gated fusion 대신 하나의 multimodal encoder 출력이 dialogue GRU/LSTM,
  memory attention, classifier 로 들어가게 했다.
- missing-modality 재임베딩: multimodal feature 가 있는 강건성 평가는 기존 feature mask만 바꾸는
  대신 `no_text`, `no_audio`, `no_video` raw sample 을 다시 만들어 extractor 를 재실행한다.
- 학습 입력 증강: `training.input_augmentation_scenarios` 로 missing-modality sample 을 train set 에
  복제하고, UID/dialogue id 를 분리해 Jina embedding 을 다시 계산할 수 있게 했다.
- cache 안정화: feature cache key 에 UID 뿐 아니라 sample modality mask digest 를 포함해, 같은
  UID라도 full/no_audio/no_video 시나리오 feature 가 섞이지 않게 했다.
- media loading 보강: extractor 가 `required_modalities` 를 선언하면 자신의 출력 modality 가
  `MULTIMODAL` 이어도 `FeaturePipeline` 이 필요한 audio/video stream 을 lazy-load 한다.
- 의존성 정리: Jina Omni 경로를 위해 `transformers>=4.57`, `torch>=2.5`, `torchvision`,
  `pillow`, `peft`, `soundfile` 조합을 optional extra 에 반영했다.

### 대표 v3pre 설정

현재 작업트리에서 fused multimodal raw MELD 실험을 실행하는 대표 설정은 다음이다.

```bash
/Users/safeailab_macmini/Desktop/2026-1-ML/configs/meld_jina_omni_dialogue_rnn.yaml
```

재현 명령은 프로젝트 루트에서 실행한다.

```bash
cd /Users/safeailab_macmini/Desktop/2026-1-ML
uv sync --extra text --extra audio --extra video --extra deep
uv run meld-emotion run --config configs/meld_jina_omni_dialogue_rnn.yaml
```

이 설정은 MELD.Raw 의 train/test CSV 와 split별 MP4 폴더를 사용하고, extractor 는 하나만 둔다.

```yaml
extractors:
  - type: jina_omni_multimodal
    model_name: jinaai/jina-embeddings-v5-omni-small
    output_dim: 1024
    batch_size: 1
    task: classification
    device: cpu
    max_video_frames: 8
```

모델은 다음처럼 `input_mode: multimodal` 로 둔다. 이때 fused embedding 차원은
`modality_encoder.text_input_dim` 에 적거나 `0` 으로 두어 train bundle 에서 자동 추론할 수 있다.

```yaml
model:
  type: dialogue_rnn
  input_mode: multimodal
  modality_encoder:
    encoder_type: rnn
    text_input_dim: 1024
  training:
    input_augmentation_scenarios: [no_text, no_audio, no_video]
```

평가는 `full`, `no_text`, `no_audio`, `no_video` 시나리오를 포함한다. multimodal feature 는 이미
foundation model 내부에서 융합된 결과이므로, missing-modality 평가는 feature matrix 일부를 0으로
가리는 방식만으로는 충분하지 않다. 그래서 runner 는 각 시나리오별로 raw sample mask 를 바꾸고
Jina Omni embedding 을 다시 추출한다.

### 구현할 때의 코드 경로

README 기준으로 새 기능을 넣을 때는 다음 순서를 따른다.

1. Protocol 을 만족하는 구현체를 해당 패키지에 추가한다. 예를 들어 새 extractor 는
   `features/` 아래에서 `BaseFeatureExtractor` 를 상속하고 `FeatureMatrix` 또는
   `SequenceFeatureMatrix` 를 반환한다.
2. 설정 dataclass 를 `src/meld_emotion/config/schema.py` 에 추가하고 registry 에 등록한다.
3. YAML 복원이 필요한 중첩 설정이면 `src/meld_emotion/config/loader.py` 에 tuple/dataclass 복원
   로직을 추가한다.
4. 구성 루트인 `src/meld_emotion/pipeline/builder.py` 에 설정에서 실제 객체를 만드는 분기를
   추가한다.
5. 실행 절차가 바뀌면 `src/meld_emotion/pipeline/runner.py`, feature 조립이 바뀌면
   `src/meld_emotion/pipeline/feature_pipeline.py`, 모델 입력 계약이 바뀌면 `models/` 쪽을
   최소 범위로 수정한다.
6. `tests/` 에 config roundtrip, builder 연결, feature shape, runner scenario/cache 동작,
   모델 smoke test 를 추가하고 `uv run python -m pytest -q` 로 회귀를 확인한다.

`ours_v3pre` 의 Jina Omni 추가도 이 경로를 따른다. 설정은 `JinaOmniMultimodalConfig`, 빌더 연결은
`build_extractor`, 모델 계약은 `DialogueRnnConfig.input_mode`, 실행 절차는
`ExperimentRunner` 의 train augmentation 과 multimodal robustness 재임베딩에 들어갔다. 관련
테스트는 `tests/test_jina_omni_multimodal.py`, `tests/test_dialogue_rnn_multimodal.py`,
`tests/test_config.py` 에 분산되어 있다.

### v3pre의 남은 후보

`ours_v3pre` 는 이름 그대로 pre-release 이며, 다음 항목은 후속 검증 또는 확장 후보로 남아 있다.

- 실제 MELD.Raw 전체에서 Jina Omni run 의 정량 결과와 v2.5 sequence/dialogue 모델 비교표 기록.
- Jina Omni fused embedding 에 대한 fine-grained XAI: 현재는 단일 fused vector 단위 설명이
  중심이고 token/audio-span/frame 내부 attribution 은 v2.5 sequence extractor 경로가 더 직접적이다.
- 실행 간 feature 재사용을 위한 `DiskFeatureCache` 실제 구현. Jina Omni 재임베딩 비용이 크므로
  v3pre 이후 우선순위가 높다.
- `device: mps`/`gpu` 의 실제 메모리 프로파일링과 `max_video_frames`, `batch_size` 권장값 정리.
- Jina Omni 모델 다운로드/remote code/Hugging Face 접근 실패에 대한 사용자 가이드 보강.
- multimodal feature 와 기존 text/audio/video feature 를 함께 쓰는 hybrid ensemble 또는 two-stage
  wrapper 실험.

## ours_v2.5

`ours_v2.5` 는 `ours_v2` 의 raw MELD + foundation embedding 실험을 세 방향으로 확장한 버전이다.
첫째, text/audio 에 머물던 raw foundation 경로에 video foundation embedding 과 sequence
extractor 를 붙였다. 둘째, 발화 단위 pooled 예측을 넘어서 token/audio-span/video-frame 단위
fine-grained XAI 를 `dialogue_rnn` 과 inference 경로에 연결했다. 셋째, 기본 early/late baseline
외에 dialogue 모델 개선, ensemble/MoE/two-stage/SVM-margin two-stage wrapper, calibration/loss 설정, suite 캐시 공유를
추가해 실험 비교와 제출용 설명력을 강화했다.

현재 코드 기준 구현 상태는 `uv run meld-emotion status` 로 확인한다. 이 문서를 갱신한 시점의
상태는 전체 68개 컴포넌트 중 REAL 61개, PLACEHOLDER 7개, UNIMPLEMENTED 0개다. REAL 구현에는
MELD CSV/metadata loader, raw MP4 audio/video lazy loader, EmbeddingGemma/Wav2Vec2 XLS-R/
TimeSformer/VideoPrism extractor, fine-grained sequence extractor, sklearn/XGBoost/MLP baseline,
dialogue-level PyTorch classifier, ensemble/MoE/two-stage/SVM-margin two-stage classifier,
suite runner, console/JSON/comparison reporter 가 포함된다. placeholder 는 TF-IDF, sentence embedding, MFCC,
visual cue, stacking combiner, disk cache, dashboard HTML rendering 이다.

### ours_v2에서 v2.5로 넘어온 변화

`ours_v2` 는 MELD.Raw CSV/MP4를 직접 읽고 `text_embeddinggemma` + `audio_wav2vec2_xlsr` 로
early/late fusion baseline 을 비교하는 것이 중심이었다. `ours_v2.5` 는 그 raw-media 기반을
유지하면서 다음 변경을 더했다.

- video foundation feature: `video_timesformer`, `video_videoprism`, `video_frame_embeddings` 를
  추가해 raw MP4 프레임도 발화 단위/프레임 단위 feature 로 쓸 수 있게 했다.
- sequence feature path: `FeatureBundle.sequence_matrices` 와 `FeatureUnit` 을 추가해 pooled
  `FeatureMatrix (n,D)` 와 sequence `SequenceFeatureMatrix (n,L,D)` 를 함께 보존한다.
- fine-grained XAI: `dialogue_finegrained_xai` 가 Integrated Gradients 와 ablation 을 사용해
  token/audio span/video frame, source utterance, modality, classifier block, embedding dimension
  중요도를 저장한다.
- inference XAI: `meld-emotion infer --xai` 와 `--xai-dashboard` 가 단일 MP4+텍스트 입력에서도
  sequence extractor 와 fine-grained XAI 를 재사용한다.
- 모델 계층 확장: `dialogue_rnn` 에 Conformer encoder 선택지, focal/class-balanced loss,
  logit adjustment, hard negative mining, calibration, neutral gate 를 추가하고, `ensemble`,
  `moe`, `two_stage` wrapper 를 등록했다.
- 평가 확장: 기존 accuracy/F1/recall 외에 `nll`, `brier_score`,
  `expected_calibration_error`, `classwise_ece`, `confidence_bucket_accuracy`,
  `high_confidence_wrong` 을 metric registry 에 추가했다.
- 실행 편의: `configs/default.yaml` 은 `two_stage` 기본 smoke run 이고, `scripts/train.py`,
  `scripts/evaluate.py`, `scripts/infer.py` wrapper 는 CLI와 같은 기능을 얇게 감싼다.
- suite cache: dataset/extractors/media/train split/eval split signature 가 같은 실험끼리는
  `InMemoryFeatureCache` 를 공유해 같은 foundation feature 를 한 suite 안에서 재사용한다.

### 대표 raw foundation/sequence 설정

현재 작업트리에서 raw MELD 세 모달리티를 모두 쓰는 대표 설정은 sequence dialogue run 이다.

```bash
/Users/safeailab_macmini/Desktop/2026-1-ML/configs/meld_sequence_dialogue_rnn.yaml
```

재현 명령은 프로젝트 루트에서 실행한다.

```bash
cd /Users/safeailab_macmini/Desktop/2026-1-ML
uv sync --extra text --extra audio --extra video --extra deep
uv run meld-emotion run --config configs/meld_sequence_dialogue_rnn.yaml
```

이 설정은 `text_token_embeddings`, `audio_wav2vec2_xlsr_sequence`, `video_frame_embeddings` 를
함께 사용하고 `dialogue_rnn` 의 Conformer modality encoder 경로를 학습한다. 평가는 `full`,
`no_text`, `no_audio`, `no_video` 시나리오를 포함하며, 최고 checkpoint 는
`outputs/meld_sequence_dialogue_rnn_best.pt` 에 저장된다. 여러 모델을 한 번에 비교하는 현재 raw
foundation suite 는 text/audio 중심의 `configs/meld_embeddinggemma_wav2vec2_suite.yaml` 이다.

### Fine-grained Dialogue XAI

`ours_v2.5` 의 fine-grained XAI 는 `dialogue_rnn` 에서 발화 단위 pooled embedding 보다 아래로
내려가 단어/token, 오디오 시간 구간, 비디오 프레임 단위 설명을 저장한다. 새 sequence extractor
는 다음 세 가지다.

| type | 역할 |
| --- | --- |
| `text_token_embeddings` | Hugging Face tokenizer/model 로 token embedding 과 character span 보존 |
| `audio_wav2vec2_xlsr_sequence` | Wav2Vec2 XLS-R `last_hidden_state` 를 시간 step별 embedding 으로 보존 |
| `video_frame_embeddings` | CLIP vision model 로 sampled frame별 embedding 생성 |

`TorchDialogueEmotionClassifier` 는 sequence feature 가 있으면 `[B,N,L,D]` 와 `[B,N,L]` mask 를
사용하고, 없으면 기존처럼 `[B,N,1,D]` 로 fallback 한다. 모델 forward 의 `return_xai=True` 경로는
`modality_gate`, encoder attention, `memory_attention`, `u_text/u_audio/u_video`, `fused`,
`context_h`, `memory` 를 반환한다. classifier head 에 들어가는 `fused/context/memory` block 은
block ablation 으로 target logit 변화량을 계산한다.

`dialogue_finegrained_xai` 는 Captum Integrated Gradients 로 target `logits[b,t,c]` attribution 을
계산하고, 같은 attribution tensor 를 다음 해상도로 집계한다.

- token/audio-span/video-frame top-k
- source utterance importance + `memory_attention`
- modality attribution share + modality logit ablation delta + `modality_gate`
- `fused/context/memory` block importance
- text/audio/video embedding dimension attribution

예제 설정은 다음이다.

```bash
uv sync --extra text --extra audio --extra video --extra deep --extra xai
uv run meld-emotion run --config configs/example_finegrained_xai.yaml
```

전체 결과는 `outputs/finegrained_xai.json`, dashboard data contract 는
`outputs/finegrained_xai_dashboard.json` 에 저장된다. 실제 HTML dashboard 렌더링은 아직
placeholder 이며, 현재 exporter 는 frontend 가 사용할 수 있는 JSON payload 를 저장한다. 자세한
해석 방법과 dashboard payload 는 `docs/finegrained_xai.md` 에 정리했다.

단일 inference 경로도 fine-grained XAI 를 지원한다. `meld-emotion infer --xai` 는 기본 pooled
extractor 대신 sequence extractor 조합을 사용하고, 감정 분류 결과와 함께 XAI 결과를 console 또는
JSON 으로 출력한다. `--xai-dashboard <path>` 를 지정하면 단일 입력용 dashboard payload 도 저장한다.

### v2.5의 남은 후보

다음 항목은 `ours_v2.5` 이후 구현하면 좋은 후속 작업이다.

- EmbeddingGemma/TimeSformer/VideoPrism 내부 token-patch representation 을 직접 노출하는 extractor.
- 비디오 영역/얼굴 부위 heatmap: patch-level attribution, Grad-CAM, face landmark/region attribution.
- raw occlusion XAI: 단어 삭제, waveform 구간 무음화, frame 제거 후 feature 재추출.
- SHAP/LIME 계열 group explanation: modality group, utterance group, token/span/frame group.
- Dashboard HTML/UI 렌더링: heatmap, timeline, dialogue graph, modality/block comparison view.
- Captum method 확장: GradientSHAP, DeepLIFT, Layer Integrated Gradients.
- Dataset mean / silence / blank frame baseline 선택 옵션.
- XAI cache: 비싼 attribution 결과를 target/model/checkpoint/config signature 기준으로 저장.
- End-to-end raw input wrapper: extractor 와 classifier 를 묶어 원 입력 기준 attribution 충실도 개선.

## ours_v2

`ours_v2` 는 MELD.Raw 의 CSV/MP4를 이 프로젝트 파이프라인 안에서 직접 읽고,
텍스트는 `google/embeddinggemma-300m`, 오디오는 `facebook/wav2vec2-xls-r-300m` 으로
foundation embedding 을 추출해 early/late fusion baseline 을 비교하는 버전이다. `ours_v1` 이
MELD 팀 제공 precomputed pickle 을 중심으로 비교했다면, `ours_v2` 는 raw MELD 경로,
split별 media folder, lazy media loading, gated text model 인증, audio model loading, raw media
오류 처리까지 실험 설정에 통합한 쪽이다.

실행 기준 파일은 다음이다.

```bash
/Users/safeailab_macmini/Desktop/2026-1-ML/configs/meld_embeddinggemma_wav2vec2_suite.yaml
```

재현 명령은 프로젝트 루트에서 실행한다.

```bash
cd /Users/safeailab_macmini/Desktop/2026-1-ML
uv sync --extra text --extra audio
uv run meld-emotion compare --config configs/meld_embeddinggemma_wav2vec2_suite.yaml
```

EmbeddingGemma 는 Hugging Face gated model 이므로 최초 실행 전 Google Gemma 라이선스 동의와
Read token 인증이 필요하다.

```bash
uv run huggingface-cli login
uv run huggingface-cli whoami
```

비대화형 환경에서는 `HF_TOKEN=hf_...` 로 같은 인증을 제공한다. Wav2Vec2 XLS-R 는 Hugging Face
Transformers 와 PyTorch 가 필요하고, raw MP4 오디오는 `MediaLoader` 가 16kHz mono waveform 으로
lazy-load 한다. 현재 장비에서는 MPS 가 사용 가능하지 않아 suite 의 두 foundation extractor 는
`device: cpu` 로 명시한다. 첫 실행은 train/test 전체 발화에 대해 두 foundation model embedding 을
계산하므로 오래 걸린다. suite 내부에서는 같은 feature signature 를 가진 실험끼리
`InMemoryFeatureCache` 를 공유하지만, `DiskFeatureCache` 는 아직 실행 간 영속화가 아닌
placeholder 다.

### 사용 데이터와 경로

`ours_v2` 는 MELD.Raw 의 CSV와 MP4를 직접 사용한다.

```text
MELD.Raw/train/train_sent_emo.csv
MELD.Raw/dev_sent_emo.csv
MELD.Raw/test_sent_emo.csv
MELD.Raw/train/train_splits/
MELD.Raw/dev_splits_complete/
MELD.Raw/output_repeated_splits_test/
```

설정의 split 은 `train_split: train`, `eval_split: test` 이다. `MeldConfig` 에
`audio_subdir_train`/`audio_subdir_dev`/`audio_subdir_test` 와
`video_subdir_train`/`video_subdir_dev`/`video_subdir_test` 를 추가해 split마다 다른 MP4 폴더를
YAML 로 지정할 수 있게 했다. 오디오와 비디오 subdir 를 모두 지정하지만, 이 suite 의 extractor 는
text/audio 만 사용하므로 실제 lazy-load 는 오디오 waveform 에 대해서만 발생한다.

MELD.Raw train split 에 PyAV 가 열 수 없는 손상 MP4 1개(`dia125_utt3.mp4`)가 있어
`media.on_error: drop_sample` 을 사용한다. media 로딩 실패 샘플은 학습/평가에서 제외되며,
러너 metadata 에는 raw 샘플 수(`n_train_raw`/`n_test_raw`)와 실제 특징화 후 샘플 수
(`n_train`/`n_test`)가 함께 기록된다.

MELD.Raw test split 의 `dia38_utt4.mp4` 는 실제 MP4/container 길이가 약 305초다. Wav2Vec2 는
self-attention 메모리가 입력 길이에 대해 제곱으로 증가하므로, 이 파일을 그대로 넣으면
55GiB 이상의 버퍼를 요청할 수 있다. `MediaConfig.max_audio_seconds` 를 추가하고 이 suite 에서는
`max_audio_seconds: 60.0` 으로 실제 MP4 길이 1분 초과 샘플을 제외한다. 제외는 sample 단위로
일어나므로 해당 audio 뿐 아니라 대응되는 text 도 학습·평가에 쓰이지 않는다.
또한 `min_audio_seconds: 0.025` 를 추가해 CSV 구간 선택 후 Wav2Vec2 convolution 최소 입력보다
짧은 샘플도 같은 방식으로 제외한다.

### 특징 추출

YAML 의 extractor 는 두 개다.

```yaml
extractors:
  - type: text_embeddinggemma
    model_name: google/embeddinggemma-300m
    output_dim: 768
    batch_size: 32
    normalize: true
    prompt_name: Classification
    device: cpu
  - type: audio_wav2vec2_xlsr
    model_name: facebook/wav2vec2-xls-r-300m
    output_dim: 1024
    batch_size: 1
    sampling_rate: 16000
    chunk_seconds: 30.0
    normalize: true
    device: cpu
```

`EmbeddingGemmaTextExtractor` 는 `sentence-transformers` 로 모델을 lazy-load 하고,
`output_dim` 128/256/512/768 Matryoshka truncation 을 지원한다. 기본 prompt key 는
대소문자를 포함해 `Classification` 이다.

`Wav2Vec2XlsrAudioExtractor` 는 `facebook/wav2vec2-xls-r-300m` 이 ASR tokenizer vocab 이 없는
base checkpoint 라는 점을 반영해 `AutoFeatureExtractor` + `Wav2Vec2Model` 로 로드한다. 마지막
hidden state 를 attention-mask aware mean pooling 한 뒤 1024차원 발화 임베딩으로 만들고,
설정 차원보다 모델 출력이 크면 앞 차원을 사용한다. 모델 로딩 실패 시 의존성, 모델 파일 접근성,
16kHz mono 입력 조건을 확인할 수 있도록 원래 예외 원인을 포함해 `RuntimeError` 를 낸다.
`chunk_seconds` 가 설정되면 긴 waveform 은 Wav2Vec2 입력 전에 여러 chunk 로 나뉘고, chunk
임베딩은 길이 가중 평균된다.

최종 early-fusion 입력 차원은 text 768 + audio 1024 = 1792다. 이 suite 에는 concept extractor 가
없으므로 `use_concepts: false` 로 설정되어 있다.

### 모델과 비교

`ours_v2` suite 는 네 개 실험을 비교한다.

| experiment | model type | base / 구조 |
| --- | --- | --- |
| `early_centroid` | early fusion | nearest centroid, temperature 1.0 |
| `early_linear_regression` | early fusion | one-vs-rest linear regression, alpha 0.001 |
| `early_logreg` | early fusion | StandardScaler + LogisticRegression, C 1.0, max_iter 1000 |
| `late_centroid` | late fusion | text/audio 별 centroid 학습 후 mean probability combiner |

Early fusion 은 `FeatureBundle.stack()` 으로 text/audio feature 를 concatenate 한 뒤 하나의
estimator 를 학습한다. Late fusion 은 모달리티별 estimator 를 따로 학습하고 `MeanCombiner` 로
확률을 평균한다. suite runner 는 일부 변형이 모델 다운로드, 인증, native library 등 외부 경계에서
실패해도 전체 비교를 멈추지 않고 해당 outcome 의 `error` 필드에 사유를 기록한다.

### 평가, 설명, 출력

평가 metric 은 다음 네 개다.

```text
accuracy
macro_f1
weighted_f1
per_class_recall
```

confusion matrix 도 저장한다. suite 비교표에는 `accuracy`, `macro_f1`, `weighted_f1` 를 표시하고,
robustness 비교 기준은 `weighted_f1` 이다. 강건성 시나리오는 다음 세 개다.

```text
full
no_text
no_audio
```

`mask_bundle` 은 제거된 모달리티의 feature matrix 를 0으로 만들고 availability 를 false 로
바꾼다. 설명기는 `modality_ablation` 하나를 켜 두었고, weighted F1 하락폭으로 text/audio
모달리티 기여도를 기록한다.

주요 결과 파일은 다음이다.

```text
outputs/meld_embeddinggemma_wav2vec2_models.json
```

파일은 `ComparisonReport` 직렬화 결과이며, 성공한 실험은 `result`, 실패한 실험은 `error` 를
담는다. 모델 접근 권한이나 네트워크/캐시 상태가 준비되지 않은 환경에서는 Wav2Vec2 또는
EmbeddingGemma 로딩 실패가 `error` 로 남을 수 있다.

### 테스트와 검증 정책

기본 회귀 테스트는 다음 명령으로 실행한다.

```bash
uv run python -m pytest -q
uv run mypy src
uv run ruff check .
```

macOS arm64 에서 PyTorch 와 XGBoost native library 가 서로 다른 OpenMP(`libomp`) 런타임을 같은
Python 프로세스에 올릴 때 segfault 가 재현되어, XGBoost native 테스트는 `xgboost_native`
pytest marker 로 기본 테스트에서 제외했다. XGBoost 검증은 별도 프로세스에서 실행한다.

```bash
uv sync --extra xgboost
uv run python -m pytest -q -m xgboost_native
```

`tests/test_meld_embeddinggemma_wav2vec2_config.py` 는 suite YAML 이 실제
`EmbeddingGemmaTextExtractor` 와 `Wav2Vec2XlsrAudioExtractor` 를 빌드하는지, MELD.Raw 의
train/dev/test media path 가 실제 파일을 가리키는지 확인한다.

### ours_v2의 명확한 한계

- video feature 는 아직 사용하지 않는다. `video_subdir_*` 는 raw MELD 경로 정합성을 위해 지정돼
  있지만 `ours_v2` extractor 목록에는 video 가 없다. video foundation embedding 은
  `ours_v2.5` 의 `meld_sequence_dialogue_rnn.yaml` 에서 sequence feature 경로로 사용한다.
- foundation embedding 계산 결과를 실행 간 영속화하는 disk cache 는 아직 placeholder 다.
- EmbeddingGemma/Wav2Vec2 모델 다운로드와 Hugging Face 접근 권한은 실행 환경에 의존한다.
- raw audio foundation embedding 은 구현됐지만 MFCC placeholder, visual cue placeholder,
  sentence embedding placeholder, TF-IDF placeholder 는 아직 실제 구현으로 교체되지 않았다.
- 이 suite 는 dialogue-level PyTorch 모델을 포함하지 않는다. dialogue context 실험은
  `configs/example_meld_dialogue_rnn.yaml` 또는 `ours_v1` 의 precomputed-feature suite 가 기준이다.

## ours_v1

`ours_v1`은 MELD Raw의 CSV/MP4를 직접 전처리해서 특징을 뽑는 버전이 아니라, MELD metadata와
MELD 팀 제공 precomputed feature pickle을 이 프로젝트의 모듈형 파이프라인에 연결해 baseline
모델을 비교하고, 별도 raw-text 설정에서 dialogue-level PyTorch 모델을 확인한 버전이다.

현재 작업트리에서 실행 가능한 대응 설정은 두 파일로 나뉜다.

```bash
/Users/safeailab_macmini/Desktop/2026-1-ML/configs/example_meld_precomputed_baselines.yaml
/Users/safeailab_macmini/Desktop/2026-1-ML/configs/example_meld_raw_train_test_suite.yaml
```

재현 명령은 프로젝트 루트에서 실행한다.

```bash
cd /Users/safeailab_macmini/Desktop/2026-1-ML
uv sync --extra text --extra deep
uv run meld-emotion compare --config configs/example_meld_precomputed_baselines.yaml
uv run meld-emotion compare --config configs/example_meld_raw_train_test_suite.yaml
```

`text` extra는 scikit-learn baseline에 필요하고, `deep` extra는 raw-text suite 의 `dialogue_rnn`
학습에 필요하다. `example_meld_raw_train_test_suite.yaml` 의 `dialogue_rnn.training.device` 는
`mps` 이므로 Apple Silicon MPS가 없거나 불안정하면 YAML에서 `device: cpu` 로 바꿔야 한다.
아래 상세 결과 표는 예전 통합 suite 실행 결과를 보존한 역사 기록이며, 현재 파일명은 위 두
설정이 기준이다.

### 사용 데이터와 경로

프로젝트 루트는 다음이다.

```text
/Users/safeailab_macmini/Desktop/2026-1-ML
```

`ours_v1`에서 실제 데이터 로딩 기준으로 쓰는 파일은 다음 세 개다.

```text
MELD.Features.Models/features/data_emotion.p
MELD.Features.Models/features/text_glove_average_emotion.pkl
MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl
```

`data_emotion.p`는 `MeldDatasetSource(metadata_path=...)`가 읽는 metadata pickle이다. YAML에는 `dataset.type: meld`만 지정되어 있고 `root`는 생략되어 기본값 `data/MELD`가 들어가지만, `metadata_path`가 설정되어 있기 때문에 CSV 로딩 경로는 사용되지 않는다. 즉 `MELD.Raw/train/train_sent_emo.csv`, `MELD.Raw/test_sent_emo.csv`, raw MP4 파일들은 이 suite 실행에서 직접 읽히지 않는다.

metadata split 매핑은 코드상 다음과 같다.

```text
Split.TRAIN -> "train"
Split.DEV   -> "val"
Split.TEST  -> "test"
```

`ours_v1` 설정은 `train_split: train`, `eval_split: test`이므로 train과 test만 학습/평가에 사용한다. 확인된 metadata 규모는 다음과 같다.

| split | samples | dialogues | label counts |
| --- | ---: | ---: | --- |
| train | 9989 | 1038 | neutral 4710, joy 1743, sadness 684, anger 1108, surprise 1205, fear 268, disgust 271 |
| val/dev | 1109 | 114 | neutral 470, joy 163, sadness 111, anger 153, surprise 150, fear 40, disgust 22 |
| test | 2610 | 280 | neutral 1256, joy 402, sadness 208, anger 345, surprise 281, fear 50, disgust 68 |

감정 클래스 인덱스 순서는 `Emotion` enum 순서다.

```text
0 neutral
1 joy
2 sadness
3 anger
4 surprise
5 fear
6 disgust
```

### RawSample 구성

`MeldDatasetSource`는 metadata row를 `RawSample`로 바꾼다.

- `uid`: `{split}:{dialogue_id}_{utterance_id}` 형식. 예: `train:0_0`
- `dialogue_id`: metadata의 `dialog`
- `utterance_id`: metadata의 `utterance`
- `text`: metadata의 `text`
- `speaker`: 빈 문자열 `""`
- `emotion`: metadata의 `y`
- `sentiment`: `None`
- `mask`: `ModalityMask.full()`
- `audio`: `AudioInput(sample_rate=16000)`, `source_path` 없음
- `video`: `VideoInput(fps=25.0)`, `source_path` 없음
- `metadata`: `source=meld_metadata`, `num_words`

중요한 제한: metadata 기반 로딩에서는 실제 화자명이 들어오지 않는다. 따라서 `dialogue_rnn`의 speaker embedding은 구조상 존재하지만, `ours_v1`에서는 모든 발화가 같은 빈 speaker로 취급된다.

### 특징 추출과 전처리

YAML의 extractor는 두 개다.

```yaml
extractors:
  - type: meld_precomputed
    path: MELD.Features.Models/features/text_glove_average_emotion.pkl
    modality: text
  - type: meld_precomputed
    path: MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl
    modality: audio
```

두 extractor 모두 `MeldPrecomputedFeatureExtractor`를 사용한다. pickle은 `(train, dev, test)` 세 split mapping을 담고 있어야 하며, 각 sample의 key는 우선 `{dialogue_id}_{utterance_id}`로 찾는다. 만약 dialogue 단위 matrix가 들어 있는 pickle이면 dialogue id로 matrix를 찾고 utterance id row를 선택하는 fallback도 있다.

확인된 feature 차원은 다음과 같다.

| feature file | modality | kind | split items | dim |
| --- | --- | --- | ---: | ---: |
| `text_glove_average_emotion.pkl` | text | embedding | train 9989, dev 1109, test 2610 | 300 |
| `audio_embeddings_feature_selection_emotion.pkl` | audio | embedding | train 9989, dev 1109, test 2610 | 1611 |

`FeaturePipeline`은 extractor별 `fit` 후 `transform`을 수행하고 `FeatureBundle`을 만든다. 이때 matrix는 sample 순서대로 `np.vstack`된다. `ours_v1`의 최종 early-fusion 입력 차원은 text 300 + audio 1611 = 1911이다.

raw MP4 lazy-load는 이 suite에서 사실상 발생하지 않는다. audio extractor가 있으므로 `FeaturePipeline`은 audio 필요 여부를 감지하지만, metadata 기반 `AudioInput`에는 `source_path`가 없고 precomputed extractor는 waveform을 요구하지 않는다. 따라서 PyAV/ffmpeg 기반 audio 로딩, frame 로딩, MFCC, 얼굴 landmark 같은 raw 전처리는 `ours_v1`에 포함되지 않는다.

캐시는 `cache.type: memory`라서 한 실행 안에서만 feature matrix를 재사용한다. 디스크 캐시 산출물은 없다.

### 모델 구성

`ours_v1` suite는 총 8개 실험을 비교한다.

| experiment | model type | base / 구조 |
| --- | --- | --- |
| `majority` | early fusion | 최빈 클래스 예측 |
| `random` | early fusion | seed 0 random probability |
| `early_centroid` | early fusion | nearest centroid, temperature 1.0 |
| `early_linear_regression` | early fusion | one-vs-rest ridge-style linear regression, alpha 0.001 |
| `early_logreg` | early fusion | StandardScaler + LogisticRegression, C 1.0, max_iter 1000 |
| `early_svm` | early fusion | StandardScaler + SVC, C 1.0, RBF kernel, probability=True |
| `late_centroid` | late fusion | modality별 centroid 학습 후 mean probability combiner |
| `dialogue_rnn` | dialogue-level classifier | PyTorch GRU + gated fusion + causal memory attention |

Early fusion은 `FeatureBundle.stack()`으로 모든 feature matrix를 concatenate한 뒤 하나의 estimator를 학습한다. `use_concepts: true`로 되어 있지만, `ours_v1` extractor는 모두 `embedding` kind라서 concept feature는 없다.

Late fusion은 모달리티별로 별도 estimator를 학습하고, 예측 시 각 모달리티 probability를 combiner가 합친다. `late_centroid`는 text centroid와 audio centroid를 따로 학습한 뒤 `MeanCombiner`를 사용한다.

### dialogue_rnn 구조

`dialogue_rnn`은 `TorchDialogueEmotionClassifier` adapter가 `FeatureBundle`을 dialogue batch tensor로 재구성한 뒤 `MultimodalEmotionModel`을 학습한다.

입력 재구성 방식:

- 발화를 `dialogue_id`별로 묶고 `utterance_id` 순서로 정렬한다.
- batch 단위는 sample이 아니라 dialogue다.
- 각 dialogue는 batch 내 max dialogue length까지 padding된다.
- `text_x`, `audio_x`, `video_x` shape는 `[B, N, 1, D]`다.
- `utterance_mask`는 padding 발화를 0으로 표시한다.
- `modality_mask`는 `[B, N, 3]`이며 text/audio/video 사용 가능 여부를 담는다.

`ours_v1`에는 video extractor가 없다. 그래서 video 입력 차원은 내부적으로 1로 보정되고 값은 0이며, `modality_mask[..., 2]`는 0이다. 모델 구조에는 video branch가 있지만 학습 신호는 text/audio에서만 온다.

모델 내부 모듈:

- `AttentiveRnnEncoder` 3개: text/audio/video 각각 `Linear(input_dim -> proj_dim)` + `LayerNorm` + dropout + GRU/LSTM + attention pooling
- `GatedMultimodalFusion`: text/audio/video embedding과 modality mask를 받아 gated sum을 만들고, interaction feature 사용 시 `text*audio`, `text*video`, `audio*video`도 concat
- `DialogueContextRnn`: fused utterance vector와 speaker embedding을 concat한 뒤 dialogue-level GRU/LSTM으로 문맥 인코딩
- `MemoryAttention`: 현재 발화가 자기 자신과 과거 발화에만 attend하는 causal attention. relative distance bias와 same-speaker bias 사용 가능
- `EmotionClassifierHead`: fused vector, context vector, memory vector를 concat해 7-class logits 출력

`ours_v1` YAML의 주요 hyperparameter:

| block | value |
| --- | --- |
| `rnn_type` | `gru` |
| modality encoder | `proj_dim=128`, `hidden_dim=128`, `dropout=0.2` |
| fusion | `fusion_dim=256`, `dropout=0.3`, gated fusion on, interaction features on |
| dialogue context | `speaker_emb_dim=32`, `hidden_dim=256`, `num_layers=1`, `dropout=0.3` |
| memory attention | enabled, `attn_dim=256`, RoPE off, relative distance bias on, same speaker bias on, max relative distance 32 |
| classifier head | `hidden_dim=256`, `dropout=0.3` |
| training | `lr=0.0002`, `weight_decay=0.01`, `gradient_clip_norm=1.0`, `batch_size=8`, `max_epochs=100`, `early_stopping_patience=10`, `validation_fraction=0.1`, `modality_dropout=0.1`, `seed=0`, `device=mps` |

학습 방식:

- optimizer: `torch.optim.AdamW`
- loss: `CrossEntropyLoss`
- class imbalance 대응: train label 빈도 기반 class weight 사용
- validation split: train dialogues를 seed 0으로 shuffle한 뒤 dialogue 단위 10%를 validation으로 사용
- early stopping 기준: validation weighted F1
- gradient clipping: norm 1.0
- training-time modality dropout: 각 발화/모달리티를 확률 0.1로 drop하되, 사용 가능한 모달리티가 모두 사라지면 첫 번째 available modality를 복구
- best validation state는 CPU state dict로 보관했다가 학습 종료 후 reload

현재 모델 checkpoint는 파일로 저장하지 않는다. 실행이 끝나면 JSON 결과만 남는다.

### 평가와 강건성 시나리오

평가 metric은 다음 네 개다.

```text
accuracy
macro_f1
weighted_f1
per_class_recall
```

confusion matrix도 저장한다. suite 비교표에는 `metrics: [accuracy, macro_f1, weighted_f1]`만 표시하고, robustness 비교 기준은 `weighted_f1`이다.

강건성 시나리오는 다음 세 개다.

```text
full
no_text
no_audio
```

`mask_bundle`은 제거된 모달리티의 feature matrix 값을 0으로 만들고 availability도 false로 바꾼다. 모델은 같은 학습된 checkpoint/estimator를 사용해 masked test bundle에서 다시 평가된다.

### 출력 구조와 결과물

주요 결과 파일은 다음이다.

```text
outputs/meld_raw_train_test_models.json
```

파일은 `ComparisonReport` 직렬화 결과다.

```text
{
  "name": "meld_raw_train_test_models",
  "outcomes": [
    {
      "name": "...",
      "result": {
        "name": "...",
        "evaluation": {
          "scenario": "full",
          "metrics": [...],
          "confusion": {"matrix": ..., "labels": [...]}
        },
        "robustness": {
          "reports": [
            {"scenario": "full", ...},
            {"scenario": "no_text", ...},
            {"scenario": "no_audio", ...}
          ]
        },
        "explanation": null,
        "metadata": {
          "classifier": "...",
          "n_train": "9989",
          "n_test": "2610",
          "train_split": "train",
          "eval_split": "test",
          "dropout": "none"
        }
      },
      "error": null
    }
  ]
}
```

suite는 일부 실험이 실패해도 전체 실행을 멈추지 않고 해당 outcome에 `error` 문자열을 기록한다. `ours_v1`의 현재 출력에서는 8개 실험이 모두 성공했다.

확인된 주요 test 성능은 다음과 같다.

| experiment | accuracy | macro_f1 | weighted_f1 |
| --- | ---: | ---: | ---: |
| majority | 0.4812 | 0.0928 | 0.3127 |
| random | 0.1414 | 0.1112 | 0.1697 |
| early_centroid | 0.2330 | 0.1762 | 0.2229 |
| early_linear_regression | 0.3747 | 0.2475 | 0.3817 |
| early_logreg | 0.2184 | 0.1863 | 0.2520 |
| early_svm | 0.5747 | 0.2839 | 0.5105 |
| late_centroid | 0.2785 | 0.2081 | 0.2907 |
| dialogue_rnn | 0.5211 | 0.3757 | 0.5403 |

강건성 weighted F1은 다음과 같다.

| experiment | full | no_text | no_audio |
| --- | ---: | ---: | ---: |
| majority | 0.3127 | 0.3127 | 0.3127 |
| random | 0.1697 | 0.1697 | 0.1697 |
| early_centroid | 0.2229 | 0.2677 | 0.0232 |
| early_linear_regression | 0.3817 | 0.3132 | 0.3657 |
| early_logreg | 0.2520 | 0.1936 | 0.3862 |
| early_svm | 0.5105 | 0.3323 | 0.0301 |
| late_centroid | 0.2907 | 0.1656 | 0.3396 |
| dialogue_rnn | 0.5403 | 0.0266 | 0.5289 |

해석 결과(`explainers`)는 YAML에서 비어 있으므로 생성되지 않는다. 즉 permutation importance, modality ablation explanation, counterfactual explanation은 `ours_v1` 결과 JSON에 포함되지 않는다. 강건성 평가는 있지만 explanation report는 없다.

### ours_v1의 명확한 한계

- raw MP4 기반 audio/video 전처리는 사용하지 않는다.
- video feature가 없다. `dialogue_rnn`의 video branch는 구조만 있고 실제 입력은 0/masked 상태다.
- metadata 로딩에서는 speaker가 빈 문자열이라 speaker-aware 모델의 장점이 제한된다.
- precomputed GloVe average와 audio embedding이 어떤 upstream 절차로 만들어졌는지는 이 프로젝트 내부에서 재현하지 않는다.
- 모델 checkpoint, per-sample prediction, attention/gate 값은 파일로 저장하지 않는다.
- suite-level `dropout`은 설정하지 않았다. `dialogue_rnn` 내부 training modality dropout만 0.1로 사용한다.
- explanation pipeline은 구현되어 있지만 `ours_v1` suite에서는 켜지 않았다.

### ours_v2 개발 참고

`ours_v2`에서 가장 자연스럽게 이어갈 수 있는 축은 다음이다.

1. video modality 추가: MELD precomputed video feature를 붙이거나 raw MP4에서 얼굴/프레임 feature를 구현해 `extractors`에 `video`를 추가한다.
2. speaker 정보 복원: metadata 대신 CSV를 기준으로 speaker를 살리되 precomputed feature와 split/key alignment를 유지하는 dataset source 또는 metadata 확장을 만든다.
3. 결과 저장 확장: model checkpoint, per-sample prediction, modality gate, memory attention, confusion matrix 요약을 별도 artifact로 저장한다.
4. explanation 활성화: `PermutationConfig`, `ModalityAblationConfig`, `CounterfactualConfig`를 suite에 추가해 해석 결과를 같이 남긴다.
5. dialogue model 튜닝: validation split을 dev split으로 분리하거나, modality dropout, class weight, fusion interaction, memory attention 옵션을 ablation한다.
6. raw feature 재현성 강화: placeholder 상태인 TF-IDF, sentence embedding, MFCC, visual cue를 실제 구현으로 대체하면 precomputed 의존도를 줄일 수 있다.
