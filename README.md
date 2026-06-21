# meld-emotion

**Concept-Guided and Multimodal Emotion Recognition on MELD** — CSE363 기말 프로젝트.

이 프로젝트는 MELD 대화 데이터의 한 발화에서 text, audio, video 정보를 읽어
`neutral`, `joy`, `sadness`, `anger`, `surprise`, `fear`, `disgust` 7개 감정을 분류하는
모듈형 실험 파이프라인이다. 단순 정확도만 보는 것이 아니라, 모달리티별 기여도, 반사실
설명, missing-modality 강건성, fine-grained XAI 까지 같이 남기는 것을 목표로 한다.

현재 최종 방향은 **foundation embedding + SVM** 이다. 초기 계획과 중간 버전에서는
`dialogue_rnn` 기반 멀티모달 RNN/Conformer 구조를 최종 모델 후보로 두었지만, 실제 구현과
비교 과정에서 SVM 계열이 더 좋은 성능을 보였다. EmbeddingGemma, Wav2Vec2 XLS-R,
TimeSformer, Jina Omni 같은 embedding 모델이 이미 정보를 잘 보존한 벡터를 만들기 때문에,
그 벡터를 RNN/fusion/context 구조로 한 번 더 압축하기보다 SVM이 embedding 공간에서 결정경계를
직접 찾는 방식이 더 적합했던 것으로 해석한다.

현재 구현 상태는 코드의 status registry 에서 직접 읽는다.

```bash
uv run meld-emotion status
```

이 문서 작성 시점 기준 상태는 전체 72개 컴포넌트 중 **REAL 65개, PLACEHOLDER 7개,
UNIMPLEMENTED 0개**다.

## 빠른 시작

가장 가벼운 smoke run 은 외부 데이터 없이 synthetic 데이터로 실행된다.

```bash
uv sync --extra dev
uv run meld-emotion run --config configs/default.yaml
uv run meld-emotion run --config configs/example_synthetic.yaml
uv run meld-emotion compare --config configs/example_suite.yaml
uv run python -m pytest -q
```

MELD raw MP4 와 foundation 모델을 쓰는 실험은 필요한 extra 를 설치한 뒤 실행한다.

```bash
uv sync --extra text --extra audio --extra video --extra deep
uv run meld-emotion run --config configs/meld_sequence_dialogue_rnn.yaml
uv run meld-emotion run --config configs/meld_jina_omni_dialogue_rnn.yaml
```

최종 v3 SVM 실험 템플릿은 `configs/test/` 아래에 있다. 이 설정은 fine-tuned
EmbeddingGemma, fine-tuned Wav2Vec2 XLS-R, original TimeSformer feature 를 early fusion 한 뒤
SVM으로 분류한다.

```bash
uv sync --extra text --extra audio --extra video
uv run meld-emotion run \
  --config configs/test/finetuned_embeddinggemma_finetuned_wav2vec2_original_timesformer_svm.yaml
```

위 설정은 다음 fine-tuned encoder 디렉터리가 미리 생성되어 있다고 가정한다.

```text
outputs/embeddinggemma_meld_finetuned/final
outputs/wav2vec2_meld_finetuned/final_encoder
```

학습된 SVM artifact 로 단일 MP4+텍스트를 추론하고 XAI 를 계산할 수 있다.

```bash
uv run meld-emotion infer \
  --mp4 sample.mp4 \
  --text "I am so happy!" \
  --checkpoint outputs/finetuned_embeddinggemma_finetuned_wav2vec2_original_timesformer_svm.pkl \
  --xai --json
```

MELD test split 전체를 저장된 SVM artifact 로 추론할 때는 batch 전용 명령을 쓴다.

```bash
uv run meld-emotion infer-svm-batch \
  --csv MELD.Raw/test_sent_emo.csv \
  --mp4-dir MELD.Raw/output_repeated_splits_test \
  --checkpoint outputs/finetuned_embeddinggemma_finetuned_wav2vec2_original_timesformer_svm.pkl
```

## 프로젝트 구조

주요 코드는 `src/meld_emotion/` 아래에 있다. 각 패키지는 `core` 의 Protocol 과 dataclass 를
통해 느슨하게 연결되고, 구체 구현 조립은 `pipeline/builder.py` 한 곳에서 한다.

| 경로 | 역할 |
| --- | --- |
| `src/meld_emotion/core/` | 공통 타입, `RawSample`, `FeatureBundle`, `PredictionSet`, Protocol, 구현 상태 마커 |
| `src/meld_emotion/config/` | YAML 과 1:1 대응되는 frozen dataclass schema, loader, suite loader |
| `src/meld_emotion/data/` | synthetic/MELD dataset source, label encoder, raw MP4 lazy media loader |
| `src/meld_emotion/features/` | text/audio/video/multimodal feature extractor |
| `src/meld_emotion/fusion/` | early fusion, late fusion, modality masking/dropout |
| `src/meld_emotion/models/` | SVM/LogReg/MLP/XGBoost/CatBoost baseline, dialogue RNN, ensemble, MoE, two-stage 계층 모델 |
| `src/meld_emotion/evaluation/` | accuracy, F1, calibration, confusion, robustness evaluation |
| `src/meld_emotion/explain/` | permutation, modality ablation, counterfactual, fine-grained dialogue XAI |
| `src/meld_emotion/reporting/` | console, JSON, comparison report, dashboard payload exporter |
| `src/meld_emotion/pipeline/` | feature cache, feature pipeline, experiment runner, suite runner, builder |
| `src/meld_emotion/fine_tunning/` | EmbeddingGemma, Wav2Vec2 XLS-R, TimeSformer fine-tuning utility |
| `configs/` | 공식 실행 YAML 과 YAML 작성 가이드 |
| `configs/test/` | 최종/비교용 실험 템플릿, SVM comparison suite |
| `docs/` | fine-grained XAI 등 상세 문서 |
| `tests/` | pytest 기반 unit, integration, config roundtrip, CLI/parser, artifact inference 테스트 |

## 실행 흐름

한 번의 실험은 다음 순서로 실행된다.

```text
DatasetSource
  -> FeaturePipeline.fit_transform(train)
  -> Classifier.fit
  -> FeaturePipeline.transform(eval)
  -> Evaluator + RobustnessEvaluator + Explainer
  -> ExperimentResult
  -> Reporter
```

YAML 로 작성한 `ExperimentConfig` 가 데이터셋, feature extractor, 모델, 평가 지표, 설명기,
리포터를 선택한다. `pipeline/builder.py` 는 설정 dataclass 를 실제 객체로 바꾸는 구성 루트다.
실행 절차 자체는 `pipeline/runner.py` 의 `ExperimentRunner.run()` 에 모여 있다.

여러 모델을 비교할 때는 `SuiteConfig` 를 사용한다.

```bash
uv run meld-emotion compare --config configs/example_suite.yaml
```

suite 는 공통 `base` 설정 위에 `experiments` 의 차이만 덮어쓴다. dataset/extractor/media/split
signature 가 같은 실험끼리는 suite 내부에서 in-memory feature cache 를 공유한다.

## 데이터와 미디어 처리

지원 데이터셋은 두 가지다.

| 데이터셋 | 설명 |
| --- | --- |
| `synthetic` | 외부 파일 없이 즉시 도는 합성 데이터. 테스트와 smoke run 용도 |
| `meld` | MELD CSV, raw MP4, 또는 MELD precomputed feature metadata 를 읽는 실제 실험 경로 |

MELD raw 경로에서는 CSV 의 `Dialogue_ID`/`Utterance_ID` 로 `dia{d}_utt{u}.mp4` 를 찾는다.
오디오와 비디오는 처음부터 모두 메모리에 올리지 않고, 해당 extractor 가 필요할 때만 lazy-load
한다. 오디오 extractor 는 waveform 만, 비디오 extractor 는 frame 만 읽는다.

`media.on_error` 로 깨진 파일 처리 방식을 고를 수 있다.

| 값 | 동작 |
| --- | --- |
| `raise` | 오류를 즉시 발생시킨다 |
| `drop_modality` | 해당 모달리티만 unavailable 로 표시한다 |
| `drop_sample` | 해당 발화를 학습/평가에서 제외한다 |

MELD.Raw train split 에 손상된 MP4 가 있을 수 있으므로, raw foundation 실험은 보통
`drop_sample` 을 사용한다.

## 특징 추출기

feature extractor 는 `FeatureMatrix(n, d)` 또는 fine-grained 용 `SequenceFeatureMatrix(n, L, d)`
를 만든다. 모달리티가 없는 샘플은 0 벡터를 반환하고, 실제 가용성은 `FeatureBundle` 의 mask 에
따로 저장된다.

| 모달리티 | 구현된 주요 extractor | placeholder |
| --- | --- | --- |
| text | `text_bow`, `text_concepts`, `text_embeddinggemma`, `text_token_embeddings` | `text_tfidf`, `text_embeddings` |
| audio | `audio_concepts`, `audio_wav2vec2_xlsr`, `audio_wav2vec2_xlsr_sequence` | `audio_mfcc` |
| video | `video_concepts`, `video_timesformer`, `video_videoprism`, `video_frame_embeddings` | `video_visual` |
| multimodal | `jina_omni_multimodal` | 없음 |
| precomputed | `meld_precomputed` | 없음 |

대표 foundation extractor 는 다음과 같다.

- `text_embeddinggemma`: `google/embeddinggemma-300m` 기반 768차원 텍스트 embedding.
- `audio_wav2vec2_xlsr`: `facebook/wav2vec2-xls-r-300m` 기반 1024차원 오디오 embedding.
- `video_timesformer`: `facebook/timesformer-base-finetuned-k400` 기반 768차원 비디오 embedding.
- `video_videoprism`: `google/videoprism-base-f16r288` 기반 비디오 embedding.
- `jina_omni_multimodal`: text/audio/video 를 Jina Omni 모델에 함께 넣는 fused multimodal embedding.

EmbeddingGemma 는 gated Hugging Face model 이므로 최초 실행 전에 라이선스 동의와 인증이 필요하다.

```bash
uv run huggingface-cli login
uv run huggingface-cli whoami
```

비대화형 환경에서는 `HF_TOKEN` 환경변수를 사용할 수 있다.

## 모델

이 프로젝트의 모델은 크게 세 층으로 나뉜다.

| 계층 | 구현 |
| --- | --- |
| Estimator | `majority`, `random`, `centroid`, `linear_regression`, `svm`, `logreg`, `random_forest`, `knn`, `xgboost`, `catboost`, `mlp` |
| Fusion classifier | `early`, `late` |
| Dialogue/compound classifier | `dialogue_rnn`, `ensemble`, `moe`, `two_stage`, `two_stage_svm_margin`, `svm_two_stage`, `svm_four_stage` |

최종 v3 기준의 중심 모델은 `early` fusion + `svm` 이다. `SvmEstimator` 는
`StandardScaler + SVC(probability=True, decision_function_shape="ovr")` 로 구현되어 있고,
sklearn 이 학습에서 본 클래스 열만 반환하는 문제를 막기 위해 전체 7-class 확률 폭으로 확장한다.

SVM 계층 모델도 구현되어 있다.

- `svm_two_stage`: SVM1 이 neutral/non-neutral 을 결정하고, SVM2 가 non-neutral 6개 감정을 분류한다.
- `svm_four_stage`: SVM1 neutral/non-neutral, SVM2 anger/joy/surprise/else, SVM3 sadness/else,
  SVM4 disgust/fear 구조로 hard routing 한다.
- `two_stage_svm_margin`: SVM margin 이 충분히 크면 SVM 예측을 쓰고, 애매한 샘플은 stage2
  classifier 로 넘긴다.

`dialogue_rnn` 도 완전 구현되어 있다. text/audio/video encoder, gated multimodal fusion,
speaker-aware dialogue RNN, causal memory attention, classifier head 를 가진다. sequence feature 를
사용하면 `[B, N, L, D]` 입력으로 token/span/frame 단위 정보를 보존하고, pooled feature 만 있으면
`[B, N, 1, D]` 로 fallback 한다. 다만 최종 선택에서는 RNN 계열이 SVM보다 안정적인 성능을 내지
못해 최종 모델에서 밀려났다.

## Fine-Tuning

`src/meld_emotion/fine_tunning/` 에는 MELD label 로 foundation encoder 를 fine-tuning 하는 유틸리티가
있다.

```bash
uv run meld-emotion fine-tune-embeddinggemma \
  --output-dir outputs/embeddinggemma_meld_finetuned

uv run meld-emotion fine-tune-wav2vec2 \
  --output-dir outputs/wav2vec2_meld_finetuned \
  --device mps

PYTORCH_ENABLE_MPS_FALLBACK=1 uv run meld-emotion fine-tune-timesformer \
  --output-dir outputs/timesformer_meld_finetuned \
  --device mps
```

최종 SVM 템플릿은 fine-tuned EmbeddingGemma 와 Wav2Vec2 encoder 를 입력 extractor 로 다시 사용한다.
TimeSformer 는 original 또는 fine-tuned encoder 조합을 `configs/test/` suite 에서 비교할 수 있다.

## 평가와 설명

평가 지표는 accuracy, macro F1, weighted F1, per-class recall 뿐 아니라 NLL, Brier score,
expected calibration error, classwise ECE, confidence bucket accuracy, high-confidence wrong 도
지원한다.

강건성 평가는 `evaluation.scenarios` 로 제어한다.

```yaml
evaluation:
  metrics: [accuracy, macro_f1, weighted_f1]
  scenarios: [full, no_text, no_audio, no_video, text_only, audio_only, video_only]
```

설명기는 네 종류가 구현되어 있다.

| 설명기 | 방식 |
| --- | --- |
| `permutation` | feature column 또는 group permutation 으로 성능 하락 측정 |
| `modality_ablation` | 특정 모달리티를 제거했을 때 metric drop 측정 |
| `counterfactual` | 입력 perturbation 으로 예측 변화 확인 |
| `dialogue_finegrained_xai` | Captum Integrated Gradients + ablation 으로 token/audio-span/frame/source-utterance/block attribution 계산 |

SVM artifact inference 에서는 model-agnostic XAI 도 제공한다. 모달리티 ablation, top feature
perturbation, text/audio/video unit ablation 을 계산해 JSON/Markdown 으로 저장할 수 있다.

Fine-grained dialogue XAI 는 sequence extractor 와 `dialogue_rnn` 조합에서 의미가 가장 크다.

```bash
uv sync --extra text --extra audio --extra video --extra deep --extra xai
uv run meld-emotion run --config configs/example_finegrained_xai.yaml
```

자세한 데이터 흐름은 `docs/finegrained_xai.md` 를 참고한다.

## 설정 파일

공식 대표 설정은 `configs/YAML_GUIDE.md` 에 정리되어 있다.

| 파일 | 용도 |
| --- | --- |
| `configs/default.yaml` | synthetic + two-stage + early centroid smoke run |
| `configs/example_synthetic.yaml` | 외부 데이터 없이 전체 파이프라인 확인 |
| `configs/example_suite.yaml` | 여러 실험을 비교하는 suite 예시 |
| `configs/meld_embeddinggemma_wav2vec2_suite.yaml` | MELD.Raw text/audio foundation embedding 비교 suite |
| `configs/meld_sequence_dialogue_rnn.yaml` | text/audio/video sequence feature + dialogue RNN |
| `configs/meld_jina_omni_dialogue_rnn.yaml` | Jina Omni fused multimodal embedding + dialogue RNN |
| `configs/example_finegrained_xai.yaml` | sequence feature 기반 fine-grained XAI 예시 |

`configs/test/` 에는 최종/비교용 실험 템플릿이 있다.

| 파일 | 용도 |
| --- | --- |
| `configs/test/finetuned_embeddinggemma_finetuned_wav2vec2_original_timesformer_svm.yaml` | v3 최종 SVM 단일 실행 템플릿 |
| `configs/test/meld_embeddinggemma_finetuned_svm_comparison.yaml` | original/fine-tuned text/audio/video encoder 조합 비교 |
| `configs/test/meld_foundation_all_models_suite.yaml` | foundation feature 위에서 여러 모델 비교 |
| `configs/test/meld_jina_omni_model_comparison.yaml` | Jina Omni embedding 위 baseline/RNN 비교 |
| `configs/test/meld_sequence_svm_hierarchy_suite.yaml` | sequence/foundation feature 기반 SVM hierarchy 비교 |

## 기존 계획에서 바뀐 점

버전별 상세 이력은 `version_history.md` 에 있다. 큰 흐름은 다음과 같다.

1. `ours_v1`: MELD precomputed feature 와 기본 early/late fusion 중심.
2. `ours_v2`: MELD.Raw CSV/MP4 를 직접 읽고 EmbeddingGemma + Wav2Vec2 XLS-R raw foundation
   feature 실험으로 확장.
3. `ours_v2.5`: TimeSformer/VideoPrism, sequence feature, dialogue RNN, fine-grained XAI,
   ensemble/MoE/two-stage/calibration 기능 추가.
4. `ours_v3pre`: Jina Omni fused multimodal embedding 과 `dialogue_rnn.input_mode: multimodal`
   경로 추가.
5. `ours_v3`: 최종 모델을 RNN based 모델에서 SVM 기반 모델로 변경. 이유는 embedding 모델이 만든
   고품질 벡터를 다시 RNN으로 압축하는 것보다, SVM이 해당 벡터 공간에서 margin 기반 분리를 하는
   방식이 현재 성능상 더 적합했기 때문이다.

즉, 초기 계획은 해석 가능한 concept-guided multimodal RNN 구조까지 가는 것이었지만, 최종 성능
판단은 더 단순한 SVM 쪽으로 이동했다. 다만 RNN, fine-grained XAI, hierarchy, Jina Omni 경로는
비교/분석용 구현으로 남아 있다.

## 아직 placeholder 인 것

`uv run meld-emotion status` 기준 미구현(`UNIMPLEMENTED`)은 0개지만, 다음 7개는 placeholder 다.

| 컴포넌트 | 현재 상태 |
| --- | --- |
| `features.text.tfidf.TfidfTextExtractor` | 실제 `TfidfVectorizer` 학습 대신 결정적 대체 특징 |
| `features.text.embeddings.SentenceEmbeddingExtractor` | sentence-transformers 일반 문장 임베딩 미구현 |
| `features.audio.acoustic.MfccAcousticExtractor` | librosa 기반 MFCC/스펙트럴/피치 통계 미구현 |
| `features.video.visual.VisualCueExtractor` | OpenCV/MediaPipe 얼굴, 랜드마크, 움직임 단서 미구현 |
| `fusion.combiners.StackingCombiner` | 실제 stacking meta learner 대신 평균 fallback |
| `pipeline.cache.DiskFeatureCache` | 실행 간 디스크 feature 재사용 미구현, 현재 memory cache 위임 |
| `reporting.report.DashboardExporter` | HTML dashboard 렌더링 미구현, JSON payload 만 저장 |

후속 작업 후보는 final comparison table 고정, SVM hyperparameter search, SVM hierarchy 정량 검증,
SVM XAI dashboard 렌더링, disk feature cache 구현이다.

## 새 기능을 추가하는 방법

이 프로젝트는 기능을 직접 if 문으로 흩뿌리지 않고, Protocol 구현체와 설정 dataclass 를 통해
추가한다.

1. `core/protocols.py` 의 계약을 만족하는 구현체를 해당 패키지에 만든다.
2. `src/meld_emotion/config/schema.py` 에 frozen dataclass 설정을 추가하고 registry 에 등록한다.
3. YAML 에서 중첩 설정으로 복원해야 하면 `src/meld_emotion/config/loader.py` 에 복원 로직을
   추가한다.
4. `src/meld_emotion/pipeline/builder.py` 의 `build_*` 함수에 설정에서 실제 객체를 만드는 분기를
   추가한다.
5. 실행 절차 자체가 바뀌면 `pipeline/runner.py`, feature 조립 방식이 바뀌면
   `pipeline/feature_pipeline.py`, 모델 입력 계약이 바뀌면 `models/` 쪽을 수정한다.
6. `tests/` 에 config roundtrip, builder 연결, shape, runner smoke, inference 또는 XAI 회귀
   테스트를 추가한다.

새 지표는 `METRIC_REGISTRY` 등록만으로 충분하고, 새 robustness scenario 는
`fusion/masking.py` 의 scenario 정의에 추가한다.

## 개발과 검증

기본 검증은 다음 명령으로 수행한다.

```bash
uv run python -m pytest -q
uv run mypy src
uv run ruff check .
```

XGBoost/CatBoost native 테스트는 macOS arm64 에서 PyTorch 와 OpenMP 런타임 충돌이 날 수 있어
기본 pytest 와 분리한다.

```bash
uv sync --extra xgboost
uv run python -m pytest -q -m xgboost_native

uv sync --extra catboost
uv run python -m pytest -q -m catboost_native
```

로그는 기본 `INFO` 로 stderr 에 출력된다.

```bash
uv run meld-emotion run --config configs/default.yaml --log-level DEBUG --log-file outputs/run.log
```

## 참고 문서

- `version_history.md`: 버전별 구현 변화와 최종 SVM 전환 배경
- `configs/YAML_GUIDE.md`: YAML 작성 규칙과 대표 설정
- `docs/finegrained_xai.md`: fine-grained dialogue XAI 데이터 흐름과 해석 주의사항
- `src/meld_emotion/*/README.md`: 패키지별 확장 가이드
