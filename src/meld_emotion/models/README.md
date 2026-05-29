# models — 기초 학습기(Estimator)

`Estimator` 는 평범한 행렬 `(X, y)` 을 받는 sklearn 형태의 단위 학습기다(`fit`/`predict`/
`predict_proba`). 융합 분류기(Early/Late)가 내부적으로 이를 감싸 사용한다. 레이블은 0..K-1
연속 정수라고 가정한다(`EmotionLabelEncoder` 보장).

## 구성

- `baselines.py` (완전 구현, numpy 전용):
  - `MajorityClassEstimator` — 최빈 클래스(성능 하한선).
  - `RandomEstimator` — 시드 기반 무작위(형상/정상성 검증).
  - `NearestCentroidEstimator` — z-표준화 후 클래스 중심 거리 분류. 합성 데이터에서 실제 학습.
- `sklearn_estimators.py` (**미구현**): `SvmEstimator`, `LogisticRegressionEstimator` — 교수님
  피드백에서 명시한 베이스라인. scikit-learn 으로 각 ~10줄 래퍼로 채운다.

## 새 학습 알고리즘 추가하기

1. `Estimator` Protocol 을 만족하는 클래스를 작성(`fit(x,y)->Self`, `predict(x)`,
   `predict_proba(x)`)하고 `@real` 태그.
2. [config/schema.py](../config/README.md) 에 `EstimatorConfig` 하위 설정 추가·등록.
3. [pipeline/builder.py](../pipeline/builder.py) `build_estimator_factory` 에 분기 추가
   (팩토리는 매 호출 **새 인스턴스**를 반환해야 한다 — Late fusion 이 모달리티마다 하나씩 학습).

## SVM/LogReg 구현 메모

```python
from sklearn.svm import SVC
self._model = SVC(C=self.C, kernel=self.kernel, probability=True).fit(x, y)
```
`predict_proba` 가 (n, K) 를 반환하도록 `probability=True` 필요. `[text]` extra 에 scikit-learn.
