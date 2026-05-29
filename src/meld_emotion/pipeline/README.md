# pipeline — 오케스트레이션과 구성 루트

데이터→특징→융합→평가→설명→리포트를 하나로 엮는 상위 계층. 오직 `core` 의 Protocol 에만
의존한다(DIP). 구체 구현 연결은 `builder.py` 한 곳에 모인다.

## 구성

- `cache.py` — `InMemoryFeatureCache`·`NullFeatureCache` (완전 구현), `DiskFeatureCache` (임시,
  인메모리로 위임). 추출-1회·재사용-N회.
- `feature_pipeline.py` — `FeaturePipeline`: 추출기들을 학습 분할로 `fit` 후 임의 분할을
  `FeatureBundle` 로 변환하고 모달리티 가용성 마스크를 구성한다.
- `runner.py` — `ExperimentRunner`: 한 실험을 끝까지 실행하고 `ExperimentResult` 반환.
- `builder.py` — **구성 루트**. `build_experiment(config) -> ExperimentRunner`. 유일하게 모든
  구체 구현을 import 하여 설정→객체로 연결한다.

## 흐름

```
source.load → feature_pipeline.fit_transform(train) → classifier.fit
            → feature_pipeline.transform(test) → evaluator/robustness/explainers
            → ExperimentResult → reporters
```

## 새 컴포넌트를 파이프라인에 연결하기

새 데이터셋/추출기/모델/결합기/설명기/캐시/리포터는 각 패키지에서 구현·설정 등록 후, 여기
`builder.py` 의 대응 `build_*` 함수에 분기를 한 줄 추가하면 끝이다. `ExperimentRunner` 와
`core` 계약은 건드리지 않는다(OCP).

## 메모

- 러너에 `ModalityDropout` 을 주입하면 학습 시 modality dropout 증강이 적용된다.
- 특징 캐시 키는 `"{extractor.name}|{split}"` — 한 실험 내 분할별 재사용을 처리한다.
