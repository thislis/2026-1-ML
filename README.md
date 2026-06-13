# meld-emotion

**Concept-Guided and Multimodal Emotion Recognition on MELD** — CSE363 기말 프로젝트.

텍스트·오디오·비디오 발화로부터 7개 감정(neutral, joy, sadness, anger, surprise, fear,
disgust)을 분류하고, **해석 가능한 개념 벡터** `c = [c_T, c_A, c_V]` 와 **모달리티 기여도·
반사실(counterfactual) 설명**, **모달리티 누락에 대한 강건성**을 함께 평가하는 모듈형
파이프라인이다.

> 이 저장소는 **아키텍처 골격에서 출발해 실제 MELD raw/precomputed 실험까지 확장된 코드**다.
> 파이프라인 전체(데이터→특징→융합→분류→평가→설명→리포트)가 합성 데이터로 즉시
> 실행/테스트되고, MELD CSV/metadata, raw MP4 lazy-load, foundation embedding, sklearn/XGBoost,
> dialogue-level PyTorch 모델, ensemble/MoE/two-stage/SVM-margin two-stage wrapper, calibration 지표도 설정으로
> 연결된다. 아직 TF-IDF, sentence embedding, MFCC, visual cue, stacking combiner, disk cache,
> dashboard rendering 은 placeholder 경계로 남아 있다.
> 현재 상태는 `uv run meld-emotion status` 로 항상 확인할 수 있다.

## 설계 목표

- **모듈성·낮은 결합도**: 모든 단계는 [core/protocols.py](src/meld_emotion/core/protocols.py) 의
  Protocol 계약으로만 통신하고, 불변 dataclass 로 데이터를 주고받는다. 데이터셋/특징/모델/
  융합 전략/하이퍼파라미터를 서로 영향 없이 교체할 수 있다(OCP/DIP).
- **정적 분석 친화**: 전 구간 타입 힌트 + `mypy --strict`. 실행하지 않고도 동작을 예측할 수
  있도록 했다.
- **개별 테스트 용이성**: 각 컴포넌트가 작고 독립적이며, 합성 데이터로 빠르게 단위 테스트된다.

## 빠른 시작

```bash
uv sync --extra dev                                    # 환경 구성 (numpy + pytest + ruff + mypy)
uv run meld-emotion run --config configs/default.yaml  # 2-stage Neutral/Emotion 기본 smoke run
uv run meld-emotion run --config configs/example_synthetic.yaml   # 전체 파이프라인 즉시 실행
uv run meld-emotion run --config configs/example_synthetic.yaml --log-level DEBUG --log-file outputs/run.log
uv run meld-emotion compare --config configs/example_suite.yaml   # 여러 실험 비교표(Early/Late 등)
uv sync --extra text --extra audio --extra video --extra deep      # 세 sequence feature + dialogue RNN 사용 시
uv run meld-emotion run --config configs/meld_sequence_dialogue_rnn.yaml
uv run meld-emotion run --config configs/meld_jina_omni_dialogue_rnn.yaml # Jina Omni fused multimodal embedding
uv sync --extra text --extra audio --extra video --extra deep --extra xai
uv run meld-emotion run --config configs/example_finegrained_xai.yaml
uv run meld-emotion infer --mp4 sample.mp4 --text "I am so happy!" --checkpoint outputs/best_model.pt
uv run meld-emotion infer --mp4 sample.mp4 --text "I am so happy!" --checkpoint outputs/best_model.pt --xai --json
uv run python infer_emotion.py --mp4 sample.mp4 --text "I am so happy!"
uv run meld-emotion status                             # 구현 상태(완료/임시/미구현) 표
uv run python -m pytest -q                                       # 단위 + end-to-end 테스트(xgboost native 제외)
uv sync --extra xgboost                                        # XGBoost native 테스트 의존성
uv run python -m pytest -q -m xgboost_native                     # xgboost native 테스트(별도 프로세스)
uv run mypy src                                        # 정적 타입 검사 (strict)
uv run ruff check .                                    # 린트
```

가이드 문서의 `scripts/*.py` 형태를 선호하면 같은 기능을 얇은 wrapper 로 실행할 수 있다.

```bash
uv run python scripts/train.py --config configs/default.yaml
uv run python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
uv run python scripts/infer.py --checkpoint outputs/best_model.pt --input sample.mp4 --text "I am so happy!" --explain --output outputs/sample_result.json --markdown-output outputs/sample_result.md
```

`configs/default.yaml` 은 합성 데이터에서 즉시 도는 경량 기본값이며, `model.type: two_stage` 로
Model 1(Neutral/Non-Neutral) 판단 뒤 Model 2(non-neutral emotion) 판단을 명시한다. 실제 MP4
foundation feature 실험은 아래 MELD/raw 설정과 `dialogue_rnn`/`two_stage.base` 조합을 사용한다.

`run`/`compare` 는 기본적으로 `INFO` 레벨 진행 로그를 stderr 로 출력한다. 더 자세히 보려면
`--log-level DEBUG`, 파일에도 남기려면 `--log-file outputs/run.log` 를 추가한다.
FeaturePipeline 은 raw media 를 chunk 단위로 읽고 최종 feature matrix 만 누적하므로, 큰 MELD MP4
실험에서도 메모리에 원본 waveform/frame 전체를 오래 붙잡지 않는다.

학습된 dialogue RNN checkpoint 로 단일 MP4+텍스트 감정을 예측하려면 `infer` 를 사용한다.
기본 checkpoint 는 `outputs/best_model.pt` 이며, 일반 inference 는 EmbeddingGemma 텍스트 임베딩,
Wav2Vec2 XLS-R 오디오 임베딩, TimeSformer 비디오 임베딩을 쓴다. `--xai` 를 켜면 BERT token,
Wav2Vec2 XLS-R sequence, CLIP frame embedding 조합으로 fine-grained XAI 를 계산한다.

```bash
uv sync --extra text --extra audio --extra video --extra deep
uv run meld-emotion infer --mp4 sample.mp4 --text "I am so happy!" --checkpoint outputs/best_model.pt
uv run meld-emotion infer --mp4 sample.mp4 --text "I am so happy!" --json
uv run meld-emotion infer --mp4 sample.mp4 --text "I am so happy!" --xai --xai-dashboard outputs/infer_xai_dashboard.json
uv run python infer_emotion.py --mp4 sample.mp4 --text "I am so happy!"
```

이 기본 경로는 EmbeddingGemma 를 로드하므로 최초 실행 전 Hugging Face 에서
`google/embeddinggemma-300m` 라이선스에 동의하고 `uv run huggingface-cli login` 또는 `HF_TOKEN`
환경변수로 인증해야 한다. 어떤 checkpoint 를 쓰든 해당 checkpoint 를 만든 extractor 구성과 입력
차원이 inference extractor 출력 차원과 일치해야 한다.
`--xai` 를 추가하면 inference 도 `text_token_embeddings`, `audio_wav2vec2_xlsr_sequence`,
`video_frame_embeddings` sequence feature 로 예측과 fine-grained XAI 를 함께 계산한다. 이 경우
`uv sync --extra text --extra audio --extra video --extra deep --extra xai` 가 필요하며, checkpoint
의 text/audio/video input dim 이 sequence extractor 출력 차원(기본 768/1024/768)과 맞아야 한다.

macOS arm64 환경에서는 PyTorch 와 XGBoost 가 서로 다른 OpenMP(`libomp`) 런타임을 같은 Python
프로세스에 올릴 때 native segfault 가 날 수 있다. 그래서 기본 pytest 는 `xgboost_native`
마커 테스트를 제외하고, XGBoost 테스트는 위처럼 별도 pytest 프로세스에서 실행한다.

텍스트 임베딩은 경량 해싱 BoW(`type: text_bow`) 외에
`google/embeddinggemma-300m` 기반 `type: text_embeddinggemma` 도 선택할 수 있다. 이 경로는
`uv sync --extra text` 가 필요하며, Hugging Face 에서 Google Gemma 사용 조건에 동의한 계정으로
모델 접근 권한을 열어 두어야 한다.

EmbeddingGemma 는 Hugging Face gated model 이므로 최초 실행 전 인증이 필요하다.

```bash
# 1) https://huggingface.co/google/embeddinggemma-300m 에서 로그인 후 Google 라이선스 동의
# 2) https://huggingface.co/settings/tokens 에서 Read token 생성
uv run huggingface-cli login
uv run huggingface-cli whoami
```

비대화형 환경에서는 `HF_TOKEN` 환경변수로 같은 인증을 제공할 수 있다.

```bash
export HF_TOKEN=hf_your_read_token_here
```

```yaml
extractors:
  - type: text_embeddinggemma
    output_dim: 768
    prompt_name: Classification
```

오디오 임베딩은 MFCC placeholder(`type: audio_mfcc`) 외에
`facebook/wav2vec2-xls-r-300m` 기반 `type: audio_wav2vec2_xlsr` 도 선택할 수 있다. 이 경로는
`uv sync --extra audio` 가 필요하며, base XLS-R checkpoint 는 ASR tokenizer 가 없어서
`AutoFeatureExtractor` + `Wav2Vec2Model` 로 로드한다. 입력 waveform 은 16kHz mono 로 맞춰야 한다
(기본 `MediaLoader` 설정은 16kHz).

```yaml
extractors:
  - type: audio_wav2vec2_xlsr
    output_dim: 1024
    sampling_rate: 16000
    chunk_seconds: 30.0
```

비디오 임베딩은 시각 통계 placeholder(`type: video_visual`) 외에
`facebook/timesformer-base-finetuned-k400` 기반 `type: video_timesformer` 도 선택할 수 있다.
이 경로는 `uv sync --extra video` 가 필요하며, Facebook Research 의 TimeSformer 를 Hugging Face
Transformers `TimesformerModel` 경로로 lazy-load 한다. 입력 프레임은 extractor 가 8프레임,
224×224 RGB, ImageNet 정규화 입력으로 맞춘 뒤 마지막 hidden state 의 CLS token(또는 설정에
따라 token 평균)을 발화 단위 embedding 으로 만든다.

```yaml
extractors:
  - type: video_timesformer
    model_name: facebook/timesformer-base-finetuned-k400
    output_dim: 768
    num_frames: 8
    frame_size: 224
    pooling: cls
    normalize: true
```

비디오 임베딩은 추가로
`google/videoprism-base-f16r288` 기반 `type: video_videoprism` 도 선택할 수 있다. 이 경로는
`uv sync --extra video` 가 필요하며, Google DeepMind VideoPrism JAX/Flax 구현으로 checkpoint 를
lazy-load 한다. 입력 프레임은 extractor 가 16프레임, 288×288 RGB, `[0,1]` 범위로 맞춘 뒤
VideoPrism patch token 을 평균 풀링해 발화 단위 embedding 으로 만든다.
VideoPrism 패키지는 upstream dependency metadata 에 `tensorflow-cpu` 를 포함하지만,
`tensorflow-cpu==2.21.0` 은 macOS arm64 wheel 이 없어 `uv sync --extra all --extra dev` 를
막을 수 있다. 이 프로젝트는 [pyproject.toml](pyproject.toml) 의 `tool.uv.dependency-metadata`
override 로 `tensorflow-cpu` 를 제외해 lock 하고, extractor 내부에서 VideoPrism 이 import 시
필요로 하는 `tensorflow.io.gfile` 만 경량 shim 으로 제공한다. 따라서 macOS arm64 에서도
`uv sync --extra all --extra dev` 가 통과한다.

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

MELD.Raw 의 train/test 를 EmbeddingGemma 텍스트 임베딩 + Wav2Vec2 XLS-R 오디오 임베딩으로
처리해 여러 모델을 비교하려면 다음 suite 를 쓴다.

```bash
uv sync --extra text --extra audio
uv run meld-emotion compare --config configs/meld_embeddinggemma_wav2vec2_suite.yaml
```

이 suite 는 MELD.Raw train split 의 손상된 MP4 1개를 `media.on_error: drop_sample` 로 처리한다.
해당 발화는 텍스트/오디오 특징 모두 학습/평가에서 제외된다.
오디오는 CSV 의 `StartTime`/`EndTime` 구간을 사용한다. 실제 MP4 파일 길이가 60초를 넘으면
Wav2Vec2 self-attention buffer 용량 부족을 피하기 위해 샘플 전체를 제외하므로, 해당 text 도
학습·평가에 쓰이지 않는다. 또한 Wav2Vec2 convolution 최소 입력 길이보다 짧은 구간도 제외한다.
비교 대상은 `early_centroid`, `early_linear_regression`, `early_logreg`, `late_centroid` 이며,
평가 시나리오는 `full`, `no_text`, `no_audio` 다. `modality_ablation` 설명기도 켜져 있어
weighted F1 기준 모달리티별 기여도를 함께 남긴다. 출력 파일은
`outputs/meld_embeddinggemma_wav2vec2_models.json` 이다.

텍스트·오디오·비디오 sequence feature 로 raw MELD dialogue 모델을 학습하려면 다음 설정을 쓴다.

```bash
uv sync --extra text --extra audio --extra video --extra deep
uv run meld-emotion run --config configs/meld_sequence_dialogue_rnn.yaml
```

이 설정은 `text_token_embeddings`, `audio_wav2vec2_xlsr_sequence`, `video_frame_embeddings` 를
함께 사용하고 `dialogue_rnn` 의 `modality_encoder.encoder_type: conformer` 경로를 학습한다.
강건성 시나리오는 `full`, `no_text`, `no_audio`, `no_video` 이며 최고 checkpoint 는
`outputs/meld_sequence_dialogue_rnn_best.pt` 에 저장된다.

## Fine-grained Dialogue XAI

`dialogue_rnn` 에서 단어/token, 오디오 시간 구간, 비디오 프레임까지 내려가는 XAI 를 보려면
sequence extractor 와 Captum 설명기를 함께 쓴다. v1 경로는 `text_token_embeddings`,
`audio_wav2vec2_xlsr_sequence`, `video_frame_embeddings` 를 사용해 `FeatureBundle.sequence_matrices`
에 `[n,L,D]` 특징을 보존하고, 모델 adapter 가 이를 `[B,N,L,D]` dialogue tensor 로 넘긴다.

```bash
uv sync --extra text --extra audio --extra video --extra deep --extra xai
uv run meld-emotion run --config configs/example_finegrained_xai.yaml
```

`dialogue_finegrained_xai` 는 target logit 기준 Integrated Gradients 와 ablation 을 사용해
token/audio-span/video-frame top-k, source utterance 중요도, modality attribution share,
fused/context/memory block 중요도, `modality_gate`, `memory_attention`, embedding dimension
attribution 을 함께 저장한다. JSON 리포트는 `outputs/finegrained_xai.json`, dashboard data contract
는 `outputs/finegrained_xai_dashboard.json` 에 저장된다. 자세한 데이터 흐름과 해석 주의사항은
[docs/finegrained_xai.md](docs/finegrained_xai.md) 를 참고한다.

실제 MELD 실험 템플릿은 raw sequence dialogue 모델용
[configs/meld_sequence_dialogue_rnn.yaml](configs/meld_sequence_dialogue_rnn.yaml) 및
foundation embedding suite 용
[configs/meld_embeddinggemma_wav2vec2_suite.yaml](configs/meld_embeddinggemma_wav2vec2_suite.yaml) 이다.
YAML 작성 규칙과 현재 유지하는 대표 설정 목록은 [configs/YAML_GUIDE.md](configs/YAML_GUIDE.md)
에 모아 두었다.
MELD CSV/metadata 로딩, SVM 계열 베이스라인, EmbeddingGemma 텍스트 임베딩,
Wav2Vec2 XLS-R 오디오 임베딩, TimeSformer/VideoPrism 비디오 임베딩은 구현되어 있고, raw MP4 는
필요한 스트림만 lazy-load 한다(오디오 extractor 는 waveform 만, 비디오 extractor 는 프레임만
적재).
아직 남은 임시 경계(TF-IDF, MFCC, 얼굴 랜드마크 등)에 도달하면 placeholder 경고로 알려준다.
`dialogue_rnn` 모델은 발화별 특징을 dialogue batch 로 재구성해 GRU/LSTM modality encoder,
gated fusion, speaker-aware dialogue GRU, causal memory attention(RoPE 기본 off)을 학습한다.

## 디렉터리 맵

| 패키지 | 역할 | 확장 가이드 |
| --- | --- | --- |
| [core/](src/meld_emotion/core/README.md) | 도메인 타입·불변 dataclass·Protocol 계약·상태 마커 | 계약 변경 시 |
| [config/](src/meld_emotion/config/README.md) | 타입 명시 설정 dataclass ↔ YAML 로더 | 새 설정 항목 |
| [data/](src/meld_emotion/data/README.md) | 데이터셋 소스·레이블 인코더·미디어 적재 | **새 데이터셋/전처리** |
| [features/](src/meld_emotion/features/README.md) | 텍스트/오디오/비디오 × 임베딩/개념 추출기·MELD precomputed 특징 | **새 특징 추출기** |
| [models/](src/meld_emotion/models/README.md) | 기초 학습기(Estimator)·dialogue-level PyTorch Classifier | **새 학습 알고리즘** |
| [fusion/](src/meld_emotion/fusion/README.md) | Early/Late fusion·결합기·모달리티 마스킹 | **새 융합/시나리오** |
| [evaluation/](src/meld_emotion/evaluation/README.md) | 지표·평가·강건성 | **새 지표/시나리오** |
| [explain/](src/meld_emotion/explain/README.md) | permutation·모달리티 ablation·반사실 | **새 설명기** |
| [pipeline/](src/meld_emotion/pipeline/README.md) | 특징 캐시·특징 파이프라인·러너·구성 루트 | 조립/오케스트레이션 |
| [reporting/](src/meld_emotion/reporting/README.md) | 콘솔/JSON/대시보드·suite 비교 리포터 | 새 출력 형식 |
| [tests/](tests/) | pytest 기반 단위·통합·스모크 테스트, Protocol/설정 roundtrip/status/문서 가드 | 새 기능 구현 시 회귀 테스트 |

## 실험 한 번의 흐름

```
DatasetSource → FeaturePipeline(추출기들) → FeatureBundle
            → Classifier(Early/Late fusion) → 학습
            → Evaluator + RobustnessEvaluator + Explainer 들
            → ExperimentResult → Reporter 들
```

설정(`ExperimentConfig`)이 위 모든 부품을 선택한다. 파이썬에서 직접 만들거나 YAML 로
기술할 수 있고, [pipeline/builder.py](src/meld_emotion/pipeline/builder.py) 가 설정을 구체
객체로 연결하는 **유일한 구성 루트**다(다른 모듈은 구체 구현을 import 하지 않는다).

세 층을 구분하면 어디를 고칠지 빨라진다:

- **무엇을 돌릴까(선언)** → `ExperimentConfig`/YAML. 최상위 변수 목록은
  [config/README.md](src/meld_emotion/config/README.md) 참고. 학습/평가 분할과 학습 시
  modality dropout, raw MP4 미디어 적재 옵션도 여기서 켠다
  (`train_split`/`eval_split`/`dropout`/`media`).
- **설정→객체 연결(조립)** → [pipeline/builder.py](src/meld_emotion/pipeline/builder.py).
- **어떤 순서로 실행할까(절차)** → [pipeline/runner.py](src/meld_emotion/pipeline/runner.py)
  `run()`. dev 기반 모델 선택·교차검증·다중 시드 같은 **절차 변경**은 여기서 한다.
- **여러 실험을 한 번에 비교** → `meld-emotion compare`(suite). 공유 `base` + 변형 목록을
  실행해 지표·강건성 비교표(콘솔+JSON)를 낸다. 일부 변형이 미구현 경계에 닿아도 나머지는
  계속 비교된다. 형식은 [pipeline/README.md](src/meld_emotion/pipeline/README.md) 참고.

## 구현 상태와 확장

- **무엇이 되어 있나**: `uv run meld-emotion status` 가 [core/status.py](src/meld_emotion/core/status.py)
  레지스트리에서 직접 읽어 REAL / PLACEHOLDER / UNIMPLEMENTED 를 출력한다. 손으로 관리하는
  목록이 아니므로 코드와 어긋나지 않는다.
  현재 상태 기준 전체 68개 컴포넌트 중 REAL 61개, PLACEHOLDER 7개, UNIMPLEMENTED 0개다.
  `MediaLoader` 는 MP4 오디오 waveform 과 비디오 프레임을 분리해서 lazy-load 한다.
  suite 실행은 같은 dataset/extractor/media signature 를 가진 실험끼리 in-memory feature cache 를
  공유한다. 다만 `DiskFeatureCache` 는 아직 실행 간 영속화가 아닌 인메모리 위임 placeholder 다.
- **무언가를 추가/교체하려면**: 해당 축의 패키지 README 의 "새 … 추가하기" 절을 따른다.
  공통 절차는 (1) Protocol 을 만족하는 클래스 작성 → (2) [config/schema.py](src/meld_emotion/config/schema.py)
  에 설정 dataclass 추가·등록 → (3) [pipeline/builder.py](src/meld_emotion/pipeline/builder.py)
  에 설정→구체 연결 한 줄 추가.
  - **예외(더 가벼움)**: 새 **지표**는 `METRIC_REGISTRY` 에 이름으로 등록만 하면 되고(빌더
    분기 불필요), 새 **강건성 시나리오**는 [fusion/masking.py](src/meld_emotion/fusion/masking.py)
    의 `SCENARIOS` 딕셔너리에 한 줄 추가하면 된다.
  - **예외(한 단계 더)**: 다른 설정을 품는 **중첩 설정**(예: `model.base`, `late.combiner`,
    `stacking.meta`)은 [config/loader.py](src/meld_emotion/config/loader.py) 에 재귀 복원도
    추가해야 YAML 에서 읽힌다.
