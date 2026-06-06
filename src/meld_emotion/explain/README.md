# explain — 설명(Explanation)

학습된 분류기에 대한 해석을 생성한다. 각 설명기는 `Explainer` 계약
(`explain(model, bundle, y_true) -> ExplanationReport`)을 만족하고, `ExplanationReport` 의 해당
필드만 채운다(러너가 병합).

## 구성

- `permutation.py` — `PermutationImportanceExplainer`: 특징 열을 섞었을 때의 지표 하락으로
  특징 기여도 측정(개념 특징의 기여 분석). 비용이 (대상 열 수 × `n_repeats`)번의 `predict`
  이므로 기본 대상은 **개념(concept) 특징만**이다. 임베딩까지 포함하려면 설정에서
  `kinds: [concept, embedding]` (또는 임베딩만) 으로 지정한다 — 단, 실제 TF-IDF 수천 차원에는
  매우 느릴 수 있다.
- `modality_contribution.py` — `ModalityAblationExplainer`: 모달리티 제거 시 성능 하락폭으로
  모달리티 기여도 측정(modality-wise contribution).
- `counterfactual.py` — `CounterfactualExplainer`: 개념 특징을 평균값으로 되돌렸을 때 예측 확률
  변화 측정(반사실 증거 제거). 원문 단어 삭제 변형은 미구현(`explain_text_deletion`).

세 설명기 모두 현재 파이프라인에서 쓰는 특징 공간 설명 경로는 완전 구현이다. 단,
`CounterfactualExplainer.explain_text_deletion` 은 토큰 삭제 후 특징 재추출 경로가 아직 없어
명시적으로 `NotImplementedError` 를 던진다.

`configs/meld_embeddinggemma_wav2vec2_suite.yaml` 은 `modality_ablation` 을 켜서 foundation
embedding 기반 raw MELD suite 에서 weighted F1 기준 모달리티 기여도를 함께 저장한다.

## 새 설명기 추가하기

1. `Explainer` Protocol 을 구현하고 `ExplanationReport` 의 적절한 필드를 채운다(필요하면 `Metric`
   을 생성자로 주입). `@real` 태그.
2. [config/schema.py](../config/README.md) 에 `ExplainerConfig` 하위 설정 추가·등록.
3. [pipeline/builder.py](../pipeline/builder.py) `build_explainer` 에 분기 추가.

## 메모

- 설명기는 `mask_bundle`(fusion)과 `FeatureBundle.select`(core)를 재사용한다.
- 분류기가 포화(거의 완벽)되면 단일 특징 permutation 중요도는 0 에 가까울 수 있다(특징 중복).
  이때는 모달리티 ablation 이 더 뚜렷한 신호를 준다.
