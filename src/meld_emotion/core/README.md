# core — 도메인 계약과 데이터 타입

파이프라인의 **공통 어휘**. 다른 내부 모듈에 의존하지 않으며(의존성 최하단), 모든 단계가
여기 정의된 타입과 Protocol 로만 통신한다.

## 구성

- `types.py` — 열거형(`Modality`, `Emotion`, `Sentiment`, `Split`, `FeatureKind`), 배열 별칭
  (`FloatArray`/`IntArray`/`BoolArray`), 표준 순서(`EMOTION_ORDER`, `MODALITY_ORDER`).
- `data.py` — 원천 입력 dataclass: `RawSample`, `AudioInput`, `VideoInput`, `ModalityMask`.
- `features.py` — `FeatureMatrix`(추출기 1개 2D 출력), `SequenceFeatureMatrix`(token/span/frame
  단위 3D 출력), `FeatureUnit`, `FeatureBundle`(분할 전체 멀티모달 묶음,
  `stack`/`embedding_matrix`/`concept_vector`/`select` 헬퍼), `StackedFeatures`/`ColumnSpec`,
  `UtteranceSpec`(dialogue_id/utterance_id/speaker 보존).
- `results.py` — 산출물: `PredictionSet`, `MetricResult`, `EvaluationReport`, `RobustnessReport`,
  설명 결과(`FeatureContribution`/`ModalityContribution`/`CounterfactualResult`), 단일 실험
  `ExperimentResult`, suite 비교용 `ExperimentOutcome`/`ComparisonReport`.
- `protocols.py` — 단계 간 계약: `DatasetSource`, `FeatureExtractor`, `Estimator`, `Classifier`,
  `Metric`, `Explainer`, `FeatureCache`, `Reporter`, `ExperimentEvaluator`, `LabelEncoder`.
- `status.py` — 구현 상태 마커(`@real`/`@placeholder`/`@unimplemented`)와 레지스트리.
  `meld-emotion status` 는 `builder` import 로 구체 컴포넌트를 로드한 뒤 이 레지스트리를
  직접 출력한다.

## 설계 메모

- 배열을 담는 dataclass 는 `eq=False` (배열의 모호한 진리값 비교 방지).
- `FeatureBundle.stack(...)` 는 항상 `MODALITY_ORDER`(text→audio→video) 순으로 결합하여
  fit/transform 간 열 순서가 동일하게 유지된다.
- `FeatureBundle.utterances` 는 기본값이 빈 튜플인 호환 필드다. `FeaturePipeline` 이 채우며,
  dialogue-level 모델이 발화 행을 `[B,N]` dialogue batch 로 재구성할 때 사용한다.
- `FeatureBundle.sequence_matrices` 는 fine-grained XAI 용 호환 필드다. 기존 2D
  `FeatureMatrix` 기반 early/late fusion 은 그대로 동작하고, `dialogue_rnn` 은 sequence 특징이
  있으면 `[B,N,L,D]` 입력을 우선 사용한다.
- `Estimator` 는 평범한 행렬(X, y)을, `Classifier` 는 `FeatureBundle` 을 다룬다(ISP).
- 현재 상태 기준 `REAL 61`, `PLACEHOLDER 7`, `UNIMPLEMENTED 0` 이며, placeholder 는 실제 사용 시
  경고를 내거나 상태표에 사유를 표시한다.
  현재 REAL 구현에는 MELD raw/metadata loader, raw media loader, EmbeddingGemma/Wav2Vec2
  XLS-R/TimeSformer/VideoPrism extractor, fine-grained sequence extractor, sklearn/XGBoost
  baseline, MLP estimator, dialogue RNN, ensemble/MoE/two-stage/SVM-margin two-stage classifier,
  suite runner, console/JSON/comparison reporter 가 포함된다.

## 바꿀 일이 생기면

새 단계 타입을 추가할 때만 `protocols.py` 에 Protocol 을 추가한다. 기존 계약 변경은 모든
구현에 파급되므로 신중히. 데이터 모양을 늘릴 땐 dataclass 에 **기본값 있는 필드**로 추가하면
하위 호환을 유지하기 쉽다.
