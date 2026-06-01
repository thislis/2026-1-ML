# evaluation — 지표와 평가

분류 지표 계산과 강건성 평가. 모두 numpy 전용 완전 구현이라 동작이 투명하다.

## 구성

- `metrics.py` — `AccuracyMetric`, `MacroF1Metric`, `WeightedF1Metric`, `PerClassRecallMetric`,
  `confusion_counts`/`build_confusion`. 이름→인스턴스 매핑 `METRIC_REGISTRY`.
- `evaluator.py` — `Evaluator`: 지표 목록으로 한 묶음을 평가해 `EvaluationReport` 생성(+혼동행렬).
- `robustness.py` — `RobustnessEvaluator`: 여러 모달리티 시나리오에 대해 `mask_bundle` 후 반복
  평가 → `RobustnessReport`(제안서의 모달리티 누락 강건성 평가).

## 새 지표 추가하기

1. `Metric` Protocol(`name`, `compute(y_true, prediction) -> MetricResult`)을 구현하고 `@real`.
2. `metrics.py` 끝에서 `METRIC_REGISTRY.add(MyMetric.name, MyMetric)`.
3. 평가 설정의 `metrics` 목록에 이름을 넣으면 끝(빌더가 자동 생성).

```python
@real
class BalancedAccuracyMetric:
    name = "balanced_accuracy"
    def compute(self, y_true, prediction): ...
METRIC_REGISTRY.add(BalancedAccuracyMetric.name, BalancedAccuracyMetric)
```

## 새 강건성 시나리오 추가하기

[fusion/masking.py](../fusion/README.md) 의 `SCENARIOS` 에 추가하면 평가 설정 `scenarios` 에서
바로 사용된다(별도 코드 불필요).
