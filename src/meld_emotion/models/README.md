# models — 기초 학습기(Estimator)

`Estimator` 는 평범한 행렬 `(X, y)` 을 받는 sklearn 형태의 단위 학습기다(`fit`/`predict`/
`predict_proba`). 융합 분류기(Early/Late)가 내부적으로 이를 감싸 사용한다. 레이블은 0..K-1
연속 정수라고 가정한다(`EmotionLabelEncoder` 보장).

## 구성

- `baselines.py` (완전 구현, numpy 전용):
  - `MajorityClassEstimator` — 최빈 클래스(성능 하한선).
  - `RandomEstimator` — 시드 기반 무작위(형상/정상성 검증).
  - `NearestCentroidEstimator` — z-표준화 후 클래스 중심 거리 분류. 합성 데이터에서 실제 학습.
- `sklearn_estimators.py` (완전 구현, `[text]` extra 필요):
  - `SvmEstimator`(`svm`), `LogisticRegressionEstimator`(`logreg`) — 교수님 피드백에서 명시한
    베이스라인.
  - `RandomForestEstimator`(`random_forest`), `KnnEstimator`(`knn`) — 비교용 추가 베이스라인.
  - 공통 기반 `_SklearnProbaEstimator` 가 (1) sklearn 지연 import(미설치 시 명확한 `ImportError`),
    (2) 스케일 민감 모델(SVM/LogReg/KNN)에 `StandardScaler` 파이프라인 적용, (3) sklearn 이
    학습에서 본 클래스 열만 내는 `predict_proba` 를 **전체 클래스 폭 K** 로 확장하는 일을 담당한다.

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

## sklearn 베이스라인 메모

- 새 sklearn 모델은 `_SklearnProbaEstimator` 를 상속하고 `_make_model()` 만 구현하면 된다
  (확률 확장·미설치 처리·`predict` 는 기반이 담당). 스케일이 필요하면 `make_pipeline(StandardScaler(), ...)`.
- `SVC` 는 `predict_proba` 를 위해 `probability=True` 필요.
- 테스트는 sklearn 미설치 시 `pytest.importorskip` 으로 skip 된다(`tests/test_sklearn_estimators.py`).
  실제 검증은 `uv sync --extra text` 후 수행.
