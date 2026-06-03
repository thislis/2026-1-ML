# data — 데이터셋 소스와 레이블

`DatasetSource` 는 분할(`train`/`dev`/`test`)을 받아 `RawSample` 을 산출한다. 모델과 특징
추출기는 데이터 파일 형식을 직접 알지 않고, 이 패키지가 원천 데이터를 파이프라인 공통 타입으로
변환한다.

## 구성

- `labels.py` — `EmotionLabelEncoder`: `EMOTION_ORDER` 기준 감정 ↔ 정수 인덱스 변환.
- `synthetic.py` — 합성 데이터셋. 테스트와 예제 실험용.
- `meld.py` — MELD CSV 또는 baseline pickle metadata(`data_emotion.p`) 기반 데이터셋 소스.
- `media.py` — raw media 적재 경계. 현재는 명시적 미구현 상태.

## Precomputed Feature Baseline

MELD 팀이 제공한 feature pickle 은 `features/precomputed.py` 의
`MeldPrecomputedFeatureExtractor` 가 읽는다. 데이터셋은 `MeldDatasetSource(metadata_path=...)` 로
`data_emotion.p` 의 발화 metadata 를 읽고, extractor 는 각 `RawSample` 의
`Dialogue_ID`/`Utterance_ID` 에 맞는 feature row 를 반환한다.
