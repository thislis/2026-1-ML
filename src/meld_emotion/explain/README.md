# explain — 설명(Explanation)

학습된 분류기에 대한 해석을 생성한다. 각 설명기는 `Explainer` 계약
(`explain(model, bundle, y_true) -> ExplanationReport`)을 만족하고, `ExplanationReport` 의 해당
필드만 채운다(러너가 병합).

## 구성 (모두 완전 구현)

- `permutation.py` — `PermutationImportanceExplainer`: 특징 열을 섞었을 때의 지표 하락으로
  특징 기여도 측정(개념 특징의 기여 분석).
- `modality_contribution.py` — `ModalityAblationExplainer`: 모달리티 제거 시 성능 하락폭으로
  모달리티 기여도 측정(modality-wise contribution).
- `counterfactual.py` — `CounterfactualExplainer`: 개념 특징을 평균값으로 되돌렸을 때 예측 확률
  변화 측정(반사실 증거 제거). 원문 단어 삭제 변형은 미구현(`explain_text_deletion`).

## 새 설명기 추가하기

1. `Explainer` Protocol 을 구현하고 `ExplanationReport` 의 적절한 필드를 채운다(필요하면 `Metric`
   을 생성자로 주입). `@real` 태그.
2. [config/schema.py](../config/README.md) 에 `ExplainerConfig` 하위 설정 추가·등록.
3. [pipeline/builder.py](../pipeline/builder.py) `build_explainer` 에 분기 추가.

## 메모

- 설명기는 `mask_bundle`(fusion)과 `FeatureBundle.select`(core)를 재사용한다.
- 분류기가 포화(거의 완벽)되면 단일 특징 permutation 중요도는 0 에 가까울 수 있다(특징 중복).
  이때는 모달리티 ablation 이 더 뚜렷한 신호를 준다.
