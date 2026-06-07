# fusion — 융합 전략과 모달리티 마스킹

Early/Late fusion 은 동일한 `Classifier` 계약(`fit(bundle,y)`/`predict`/`predict_proba`)을
만족하므로 같은 자리에 교체 투입할 수 있다(교수님 피드백: Early/Late 비교·분석).

## 구성

- `early.py` — `EarlyFusionClassifier` (완전 구현): 모든 모달리티 특징을 한 설계 행렬로 결합 후
  단일 기초 학습기. `use_concepts` 로 개념 특징 포함 여부 조절.
- `late.py` — `LateFusionClassifier` (완전 구현): 모달리티마다 학습기 1개 → 확률을 결합기로 합침.
- `combiners.py` — `MeanCombiner`·`WeightedCombiner` (완전 구현), `StackingCombiner` (임시).
  결합 시 모달리티 **가용성**을 가중치에 곱해 누락 모달리티를 자동 제외.
- `masking.py` — `ModalityScenario` + 사전 정의 시나리오(`full`, `text_only`, `no_text`, ...),
  `mask_bundle`(시나리오에 맞춰 특징 0·가용성 False), `ModalityDropout`(학습 시 증강, 완전 구현).
  현재 기본 시나리오는 `full`, `text_only`, `audio_only`, `video_only`, `no_text`, `no_audio`,
  `no_video` 다.

## 새 융합/결합기/시나리오 추가하기

- **융합**: `Classifier` 를 구현 → `ModelConfig` 하위 설정 추가·등록 →
  [pipeline/builder.py](../pipeline/builder.py) `build_classifier` 에 분기.
- **결합기**: `ProbabilityCombiner`(`fit`/`combine`)를 구현 → `CombinerConfig` 추가·등록 →
  `build_combiner` 에 분기.
- **시나리오**: `masking.py` 의 `SCENARIOS` 딕셔너리에 `ModalityScenario` 한 줄 추가하면
  평가 설정의 `scenarios` 목록에서 바로 쓸 수 있다.

## 메모

- Late fusion 의 기초 학습기 팩토리는 모달리티 수만큼 호출되므로 매번 새 인스턴스를 반환해야 한다.
- Early fusion 은 `FeatureBundle.stack()` 의 표준 모달리티 순서(text→audio→video)를 사용하므로
  fit/transform 의 열 순서가 안정적이다. `use_concepts: false` 로 embedding 만 학습 입력에 넣을
  수 있다.
- Late fusion 은 사용 가능한 모달리티별 feature matrix 가 있는 경우에만 해당 estimator 를
  학습하고, 결합 단계에서는 availability mask 로 missing modality 확률을 제외한다.
- 강건성 평가(evaluation/robustness)는 `mask_bundle` 을 그대로 재사용한다.
- `ModalityDropout` 은 학습 bundle 에만 적용된다. 평가 시 누락 실험은 `evaluation.scenarios` 로
  지정한 `ModalityScenario` 가 담당한다.
