# reporting — 결과 리포팅

`Reporter` 는 `ExperimentResult` 를 외부로 내보낸다(`save(result)`). 여러 리포터를 동시에 둘 수
있다. suite 비교 결과(`ComparisonReport`)는 단일 실험 리포터와 입력 타입이 달라 별도의
`ComparisonReporter` 가 처리한다.

## 구성

- `report.py`:
  - `ConsoleReporter` (완전 구현): 평가·강건성·기여도 요약을 콘솔에 출력.
  - `JsonReporter` (완전 구현): 결과 전체를 JSON 으로 저장(numpy/enum 직렬화 처리).
  - `DashboardExporter` (임시): 제안서의 case-study 대시보드용 JSON 데이터 구조를 내보냄(실제
    시각화 렌더링은 미구현).
  - `ComparisonReporter` (완전 구현): `meld-emotion compare` 의 suite 결과를 콘솔 표와 JSON 으로
    저장. `Reporter` Protocol 에 주입되는 리포터는 아니며 `ComparisonReport` 를 직접 받는다.

## 새 출력 형식 추가하기

1. `Reporter` Protocol(`save(result) -> None`)을 구현하고 `@real`/`@placeholder` 태그.
2. [config/schema.py](../config/README.md) 에 `ReporterConfig` 하위 설정 추가·등록.
3. [pipeline/builder.py](../pipeline/builder.py) `build_reporter` 에 분기 추가.

## 메모

- JSON 직렬화 헬퍼 `_jsonable` 은 dataclass/numpy 배열/Enum/매핑을 재귀 변환한다. 새 결과
  타입을 추가해도 대개 그대로 직렬화된다.
- 출력 경로는 각 리포터 설정의 `path` 가 결정한다. `ExperimentConfig.output_dir` 은 현재
  리포터 경로에 자동 반영되지 않으므로, 저장 위치는 리포터의 `path` 에 직접 적는다.
- suite 비교 출력 경로는 `SuiteConfig.output_path` 가 결정한다.
