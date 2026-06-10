# reporting — 결과 리포팅

`Reporter` 는 `ExperimentResult` 를 외부로 내보낸다(`save(result)`). 여러 리포터를 동시에 둘 수
있다. suite 비교 결과(`ComparisonReport`)는 단일 실험 리포터와 입력 타입이 달라 별도의
`ComparisonReporter` 가 처리한다.

## 구성

- `report.py`:
  - `ConsoleReporter` (완전 구현): 평가·강건성·기여도 요약을 콘솔에 출력.
  - `JsonReporter` (완전 구현): 결과 전체를 JSON 으로 저장(numpy/enum 직렬화 처리).
  - `DashboardExporter` (임시): 제안서의 case-study 대시보드용 JSON 데이터 구조를 내보냄(실제
    시각화 렌더링은 미구현). fine-grained XAI 결과가 있으면 target utterance, modality,
    dialogue, block, text/audio/video, dimension panel 용 payload 를 함께 저장한다.
  - `ComparisonReporter` (완전 구현): `meld-emotion compare` 의 suite 결과를 콘솔 표와 JSON 으로
    저장. `Reporter` Protocol 에 주입되는 리포터는 아니며 `ComparisonReport` 를 직접 받는다.
    실패한 변형은 `[Failed]` 표와 JSON 의 `error` 필드에 남기고, 성공한 변형의 metric/robustness
    표는 계속 출력한다.
    모델 다운로드/인증/native library 오류처럼 특정 변형에서만 발생한 예외도 비교 전체를
    중단하지 않고 outcome 단위 실패로 기록한다.

## 새 출력 형식 추가하기

1. `Reporter` Protocol(`save(result) -> None`)을 구현하고 `@real`/`@placeholder` 태그.
2. [config/schema.py](../config/README.md) 에 `ReporterConfig` 하위 설정 추가·등록.
3. [pipeline/builder.py](../pipeline/builder.py) `build_reporter` 에 분기 추가.

## 메모

- JSON 직렬화 헬퍼 `_jsonable` 은 dataclass/numpy 배열/Enum/매핑을 재귀 변환한다. 새 결과
  타입을 추가해도 대개 그대로 직렬화된다.
- suite 결과는 성공한 변형과 실패한 변형을 같은 `ComparisonReport.outcomes` 안에 담는다.
  실패 사유는 `error` 문자열로 보존되므로, gated model 인증이나 native dependency 문제를
  결과 JSON 에서 확인할 수 있다.
- `ConsoleReporter` 는 `ExplanationReport.dialogue_xai` 가 있으면 각 target 의 top modality,
  source utterance, token/audio span/video frame 을 짧게 출력한다. 전체 값은 JSON/dashboard
  리포트에서 확인한다.
- 출력 경로는 각 리포터 설정의 `path` 가 결정한다. `ExperimentConfig.output_dir` 은 현재
  리포터 경로에 자동 반영되지 않으므로, 저장 위치는 리포터의 `path` 에 직접 적는다.
- suite 비교 출력 경로는 `SuiteConfig.output_path` 가 결정한다.
- `DashboardExporter` 는 현재 실제 UI 렌더링이 아니라 case-study 대시보드용 JSON 구조를
  저장하는 placeholder 이다.
