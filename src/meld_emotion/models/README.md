# models — 기초 학습기(Estimator)와 dialogue 모델

`Estimator` 는 평범한 행렬 `(X, y)` 을 받는 sklearn 형태의 단위 학습기다(`fit`/`predict`/
`predict_proba`). 융합 분류기(Early/Late)가 내부적으로 이를 감싸 사용한다. 레이블은 0..K-1
연속 정수라고 가정한다(`EmotionLabelEncoder` 보장).

`dialogue_rnn` 은 예외적으로 Estimator 가 아니라 `Classifier` 구현이다. 기존 발화별
`FeatureBundle` 을 dialogue batch 로 재구성한 뒤 PyTorch 모델을 직접 학습한다.
모든 `Classifier` 구현은 `EmotionLabelEncoder.classes` 의 7개 감정 순서를 기준으로 확률 폭을
고정한다.

## 구성

- `baselines.py` (완전 구현, numpy 전용):
  - `MajorityClassEstimator` — 최빈 클래스(성능 하한선).
  - `RandomEstimator` — 시드 기반 무작위(형상/정상성 검증).
  - `NearestCentroidEstimator` — z-표준화 후 클래스 중심 거리 분류. 합성 데이터에서 실제 학습.
  - `LinearRegressionEstimator`(`linear_regression`) — one-vs-rest 선형회귀 + softmax.
- `sklearn_estimators.py` (완전 구현, `[text]` extra 필요):
  - `SvmEstimator`(`svm`), `LogisticRegressionEstimator`(`logreg`) — 교수님 피드백에서 명시한
    베이스라인.
  - `RandomForestEstimator`(`random_forest`), `KnnEstimator`(`knn`) — 비교용 추가 베이스라인.
  - 공통 기반 `_SklearnProbaEstimator` 가 (1) sklearn 지연 import(미설치 시 명확한 `ImportError`),
    (2) 스케일 민감 모델(SVM/LogReg/KNN)에 `StandardScaler` 파이프라인 적용, (3) sklearn 이
    학습에서 본 클래스 열만 내는 `predict_proba` 를 **전체 클래스 폭 K** 로 확장하는 일을 담당한다.
- `xgboost_estimators.py` (완전 구현, `[xgboost]` extra 필요):
  - `XGBoostEstimator`(`xgboost`) — `XGBClassifier(objective="multi:softprob")` baseline.
- Dialogue RNN (완전 구현, `[deep]` extra 필요):
  - `attentive_rnn_encoder.py` — 모달리티별 Linear/LayerNorm/Dropout → GRU/LSTM →
    safe masked attention pooling.
  - `gated_multimodal_fusion.py` — missing-modality-aware gate, gated sum, interaction feature.
  - `dialogue_context_rnn.py` — speaker embedding 을 붙인 unidirectional dialogue GRU/LSTM.
  - `memory_attention.py` / `rope.py` — causal memory attention, padding mask, relative distance
    bias, same-speaker bias, optional RoPE(q/k only, 기본 off).
  - `multimodal_emotion_model.py` — 위 부품을 묶어 `logits`, modality gate, 각 attention 을 반환.
  - `dialogue_rnn.py` — `Classifier` adapter. `model: {type: dialogue_rnn}` 으로 선택한다.

현재 `configs/all_model_w_all_features.yaml` 은 text/audio/video foundation embedding 위에서
`majority`, `random`, early-fusion baseline, `dialogue_rnn` 을 한 suite 로 비교하는 예시다.
이 suite 의 `dialogue_rnn.training.best_checkpoint_path` 는 `outputs/best_model.pt` 로 설정되어
있어 가장 좋은 epoch 의 `model_state_dict`, 설정, speaker vocabulary, feature 차원을 함께 저장한다.
저장된 checkpoint 는 `TorchDialogueEmotionClassifier.from_checkpoint()` 로 복원되며,
`meld-emotion infer --mp4 <path> --text <text>` 와 루트 `infer_emotion.py` 가 이 경로를 사용한다.

## 클래스 수(K)는 데이터가 아니라 레이블 공간에서

모든 학습기는 생성자 첫 인자로 `n_classes` 를 받는다. 한 분할에 소수 클래스(fear/disgust)가
누락돼도 `predict_proba` 가 `(n, K)` 를 유지하도록, 융합 분류기가 인코더의 클래스 수를
팩토리에 주입한다. `y.max()+1` 추론에만 의존하면 누락 클래스에서 열 수가 어긋나 `PredictionSet`
형상 검증이 깨진다.

## 새 학습 알고리즘 추가하기

1. `Estimator` Protocol 을 만족하는 클래스를 작성(`fit(x,y)->Self`, `predict(x)`,
   `predict_proba(x)`)하고 `@real` 태그.
2. [config/schema.py](../config/README.md) 에 `EstimatorConfig` 하위 설정 추가·등록.
3. [pipeline/builder.py](../pipeline/builder.py) `build_estimator_factory` 에 분기 추가
   (팩토리는 매 호출 **새 인스턴스**를 반환해야 한다 — Late fusion 이 모달리티마다 하나씩 학습).

## Dialogue RNN 설정 메모

- 설치: `uv sync --extra deep`.
- 예제: `configs/example_meld_dialogue_rnn.yaml`.
- 기존 pooled feature extractor 출력은 발화별 single vector 이므로 wrapper 가 sequence length 1인
  `[B,N,1,D]` 로 변환한다. `SequenceFeatureMatrix` 를 내는 fine-grained extractor 가 있으면
  wrapper 는 `[B,N,L,D]` 와 `[B,N,L]` mask 를 그대로 사용한다.
- `modality_encoder.*_input_dim` 은 기본 `0` 이며, 이때 train bundle 의 실제 특징 차원을 자동
  추론한다. 값을 명시하면 실제 차원과 일치해야 한다.
- 기본 구조는 Text/Audio/Video GRU encoder → attention pooling → gated fusion → speaker
  embedding + dialogue GRU → causal memory attention(relative distance/same-speaker bias,
  RoPE off) → utterance classifier 이다.
- 입력 feature 차원이 설정값과 다르면 adapter 가 train bundle 기준으로 자동 추론하거나,
  명시 차원과 실제 차원이 충돌할 때 오류를 낸다.
- `return_xai=True` forward 는 `modality_gate`, encoder attention, `memory_attention`,
  `u_text/u_audio/u_video`, `fused`, `context_h`, `memory` 를 반환한다. `dialogue_finegrained_xai`
  설명기는 이 값들과 Captum attribution, block ablation 을 함께 저장한다.
- `training.best_checkpoint_path` 를 지정하면 weighted F1 기준 최고 모델을 저장한다. 추론 시에는
  checkpoint 의 `dims` 와 입력 extractor 출력 차원이 같아야 한다.

## sklearn 베이스라인 메모

- 새 sklearn 모델은 `_SklearnProbaEstimator` 를 상속하고 `_make_model()` 만 구현하면 된다
  (확률 확장·미설치 처리·`predict` 는 기반이 담당). 스케일이 필요하면 `make_pipeline(StandardScaler(), ...)`.
- `SVC` 는 `predict_proba` 를 위해 `probability=True` 필요.
- 테스트는 sklearn 미설치 시 `pytest.importorskip` 으로 skip 된다(`tests/test_sklearn_estimators.py`).
  실제 검증은 `uv sync --extra text` 후 수행.

## XGBoost 테스트 메모

- `tests/test_xgboost_estimator.py` 는 `xgboost_native` 마커가 붙어 기본 pytest 실행에서 제외된다.
- 실행 전 `uv sync --extra xgboost` 또는 `uv sync --extra all` 로 native 의존성을 설치한다.
- macOS arm64 에서 PyTorch 가 먼저 import 된 뒤 XGBoost native library 가 학습을 시작하면 서로
  다른 OpenMP(`libomp`) 런타임 충돌로 segfault 가 날 수 있다. 기본 회귀 테스트와 같은
  프로세스에서 섞지 않고 별도 pytest 프로세스로 실행한다.
- 검증 명령:

  ```bash
  uv run python -m pytest -q
  uv run python -m pytest -q -m xgboost_native
  ```
