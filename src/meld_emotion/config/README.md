# config — 실험 설정

타입이 명시된 **설정 dataclass** 가 단일 진실 공급원이다. 파이썬에서 직접 생성하면 mypy/IDE
검사를 온전히 받고, 동일한 구조를 YAML 로 읽고 쓸 수도 있다(하이브리드 방식).

## 구성

- `schema.py` — 모든 설정 dataclass 와 카테고리 레지스트리(`DATASET_CONFIGS`,
  `EXTRACTOR_CONFIGS`, `ESTIMATOR_CONFIGS`, `MODEL_CONFIGS`, `EXPLAINER_CONFIGS`, ...).
  각 설정은 다형성 식별자 `type` 을 `ClassVar` 로 가진다(생성자 인자가 아니라 YAML 경계 전용).
- `loader.py` — `load_config`/`dump_config`(파일), `from_dict`/`to_dict`(직렬화). `type` 을 읽어
  레지스트리에서 알맞은 dataclass 를 복원하며 중첩 설정(model.base, late.combiner,
  stacking.meta)도 재귀 처리한다.

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
