# evaluation — 지표와 평가

분류 지표 계산과 강건성 평가. 모두 numpy 전용 완전 구현이라 동작이 투명하다.

## 구성

- `metrics.py` — `AccuracyMetric`, `MacroF1Metric`, `WeightedF1Metric`, `PerClassRecallMetric`,
  `NLLMetric`, `BrierScoreMetric`, ECE 계열 calibration 지표,
  `HighConfidenceWrongMetric`, `confusion_counts`/`build_confusion`.
  이름→인스턴스 매핑 `METRIC_REGISTRY`.
- `evaluator.py` — `Evaluator`: 지표 목록으로 한 묶음을 평가해 `EvaluationReport` 생성(+혼동행렬).
  평가 시작/결과는 `INFO` 로그로 남긴다.
- `robustness.py` — `RobustnessEvaluator`: 여러 모달리티 시나리오에 대해 `mask_bundle` 후 반복
  평가 → `RobustnessReport`(제안서의 모달리티 누락 강건성 평가).
  suite 비교표에서 사용할 대표 강건성 지표는 `SuiteConfig.robustness_metric` 이 고른다.
  각 scenario 도 로그에 별도로 기록된다.

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

## 현재 기본 평가

`ExperimentConfig.evaluation` 기본 지표는 `accuracy`, `macro_f1`, `weighted_f1`,
`per_class_recall` 이고, 혼동행렬은 기본으로 저장된다. 추가로 `nll`, `brier_score`,
`expected_calibration_error`, `classwise_ece`, `confidence_bucket_accuracy`,
`high_confidence_wrong` 을 설정에서 선택할 수 있다. 기본 강건성 시나리오는 `full` 하나다. suite
설정에서는 `metrics` 로 비교표에 표시할 scalar metric 을, `robustness_metric` 으로 강건성 표의
대표 metric 을 별도로 고른다.
