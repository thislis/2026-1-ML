# config — 실험 설정

타입이 명시된 **설정 dataclass** 가 단일 진실 공급원이다. 파이썬에서 직접 생성하면 mypy/IDE
검사를 온전히 받고, 동일한 구조를 YAML 로 읽고 쓸 수도 있다(하이브리드 방식).

## 한 실험을 기술하는 것: `ExperimentConfig`

한 실험 = 하나의 `ExperimentConfig`. 아래가 최상위 필드 전부이며, 이것이 곧 "바꿀 수 있는
실험 변수" 목록이다. 절차(단계 순서) 자체는 설정이 아니라 [runner.py](../pipeline/runner.py)
`ExperimentRunner.run()` 에 있다.

| 필드 | 역할 | 기본값 |
| --- | --- | --- |
| `name` | 실험 이름(리포트·결과에 기록) | `"experiment"` |
| `seed` | 의도된 전역 시드. **현재는 각 컴포넌트의 자체 시드를 쓰며 자동 전파되지 않음** | `0` |
| `output_dir` | 출력 디렉터리(관례용). **리포터 경로에 자동 반영되지 않음** | `"outputs"` |
| `train_split` | 학습 분할 | `"train"` |
| `eval_split` | 평가·강건성·설명을 수행할 분할(예: `"dev"`) | `"test"` |
| `dataset` | 데이터셋 소스(`synthetic`/`meld`) | `synthetic` |
| `extractors` | 특징 추출기 목록(모달리티 × 임베딩/개념) | `(text_concepts,)` |
| `model` | 융합 분류기(`early`/`late`, 내부 `base` 학습기) | `early` |
| `dropout` | 학습 시 modality dropout(`None` = 미적용) | `None` |
| `media` | raw MP4 비디오 lazy-load(`video_max_frames`, `video_frame_size`) | 32프레임, 64×64 |
| `evaluation` | 지표·혼동행렬·강건성 시나리오 | 기본 4지표, `full` |
| `explainers` | 설명기 목록(permutation/ablation/counterfactual) | `()` |
| `cache` | 특징 캐시(`memory`/`null`/`disk`) | `memory` |
| `reporters` | 리포터 목록(`console`/`json`/`dashboard`) | `(console,)` |

> `seed`/`output_dir` 은 현재 장식적(decorative)이다. 재현성은 `dataset.seed`,
> `dropout.seed`, `explainer.seed` 등 컴포넌트별 시드가 책임지고, 출력 경로는 각 리포터의
> `path` 가 결정한다.

## 구성

- `schema.py` — 모든 설정 dataclass 와 카테고리 레지스트리(`DATASET_CONFIGS`,
  `EXTRACTOR_CONFIGS`, `ESTIMATOR_CONFIGS`, `MODEL_CONFIGS`, `EXPLAINER_CONFIGS`, ...).
  각 설정은 다형성 식별자 `type` 을 `ClassVar` 로 가진다(생성자 인자가 아니라 YAML 경계 전용).
- `loader.py` — `load_config`/`dump_config`(파일), `from_dict`/`to_dict`(직렬화). `type` 을 읽어
  레지스트리에서 알맞은 dataclass 를 복원하며 중첩 설정(model.base, late.combiner,
  stacking.meta)도 재귀 처리한다. 다중 실험 비교는 `SuiteConfig` + `load_suite`/`suite_from_dict`
  (공유 `base` + 변형 목록 깊은 병합) — 형식은 [pipeline/README.md](../pipeline/README.md).

## 여러 실험을 기술하는 것: `SuiteConfig`

`meld-emotion compare` 는 `SuiteConfig` 를 읽어 여러 `ExperimentConfig` 를 같은 실행 경로로
돌린다. suite YAML 의 최상위 필드는 `name`, `metrics`, `robustness_metric`, `output_path`,
공유 설정 `base`, 변형 목록 `experiments` 이다. `base` 와 각 변형은 깊은 병합되지만,
`extractors` 같은 리스트는 통째로 교체되고, 다형성 `type` 이 바뀌는 매핑은 이전 타입의 전용
필드를 끌고 오지 않도록 통째로 교체된다.

## 새 설정 항목 추가하기

1. `schema.py` 에 해당 베이스를 상속한 frozen dataclass 를 만들고 `type: ClassVar[str]` 지정.
2. 같은 파일에서 카테고리 레지스트리에 등록: `XXX_CONFIGS.add(MyConfig.type, MyConfig)`.
3. 그 설정을 구체 객체로 바꾸는 연결을 [pipeline/builder.py](../pipeline/builder.py) 에 추가.

```python
@dataclass(frozen=True)
class MyExtractorConfig(ExtractorConfig):
    type: ClassVar[str] = "text_myfeat"
    dim: int = 64
EXTRACTOR_CONFIGS.add(MyExtractorConfig.type, MyExtractorConfig)
```

YAML 에서는 `{type: text_myfeat, dim: 64}` 로 사용한다.

## 주의

- `ClassVar` 식별자는 필드 순서/기본값 문제를 피하려는 의도다(중첩 기본값은 `default_factory`).
- 새 스칼라 필드는 **기본값**을 주어야 기존 YAML 과 호환된다.
- 새 중첩 설정을 YAML 에서 복원해야 한다면 `loader.py` 에 재귀 복원 함수를 추가해야 한다
  (`model.base`, `late.combiner`, `stacking.meta`, `media.video_frame_size` 가 현재 예시다).
