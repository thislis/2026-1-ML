# meld-emotion

**Concept-Guided and Multimodal Emotion Recognition on MELD** — CSE363 기말 프로젝트.

텍스트·오디오·비디오 발화로부터 7개 감정(neutral, joy, sadness, anger, surprise, fear,
disgust)을 분류하고, **해석 가능한 개념 벡터** `c = [c_T, c_A, c_V]` 와 **모달리티 기여도·
반사실(counterfactual) 설명**, **모달리티 누락에 대한 강건성**을 함께 평가하는 모듈형
파이프라인이다.

> 이 저장소는 **아키텍처 골격**이다. 파이프라인 전체(데이터→특징→융합→분류→평가→설명→
> 리포트)가 합성 데이터로 실제 실행/테스트되며, 무거운 실제 구현(MELD I/O, TF-IDF, MFCC,
> 얼굴 랜드마크, scikit-learn 학습)은 의도적으로 임시(placeholder) 또는 미구현으로 남겨
> 두었다. 현재 상태는 `uv run meld-emotion status` 로 항상 확인할 수 있다.

## 설계 목표

- **모듈성·낮은 결합도**: 모든 단계는 [core/protocols.py](src/meld_emotion/core/protocols.py) 의
  Protocol 계약으로만 통신하고, 불변 dataclass 로 데이터를 주고받는다. 데이터셋/특징/모델/
  융합 전략/하이퍼파라미터를 서로 영향 없이 교체할 수 있다(OCP/DIP).
- **정적 분석 친화**: 전 구간 타입 힌트 + `mypy --strict`. 실행하지 않고도 동작을 예측할 수
  있도록 했다.
- **개별 테스트 용이성**: 각 컴포넌트가 작고 독립적이며, 합성 데이터로 빠르게 단위 테스트된다.

## 빠른 시작

```bash
uv sync --extra dev                                    # 환경 구성 (numpy + pytest + ruff + mypy)
uv run meld-emotion run --config configs/example_synthetic.yaml   # 전체 파이프라인 즉시 실행
uv run meld-emotion compare --config configs/example_suite.yaml   # 여러 실험 비교표(Early/Late 등)
uv run meld-emotion status                             # 구현 상태(완료/임시/미구현) 표
uv run python -m pytest -q                                       # 단위 + end-to-end 테스트
uv run mypy src                                        # 정적 타입 검사 (strict)
uv run ruff check .                                    # 린트
```

실제 MELD 실험 템플릿은 [configs/example_meld_early_svm.yaml](configs/example_meld_early_svm.yaml)
이며, 미구현 경계(MELD I/O, SVM 학습)에 도달하면 명확한 예외로 멈춘다 — 이것이 곧 구현
백로그다.

## 디렉터리 맵

| 패키지 | 역할 | 확장 가이드 |
| --- | --- | --- |
| [core/](src/meld_emotion/core/README.md) | 도메인 타입·불변 dataclass·Protocol 계약·상태 마커 | 계약 변경 시 |
| [config/](src/meld_emotion/config/README.md) | 타입 명시 설정 dataclass ↔ YAML 로더 | 새 설정 항목 |
| [data/](src/meld_emotion/data/README.md) | 데이터셋 소스·레이블 인코더·미디어 적재 | **새 데이터셋/전처리** |
| [features/](src/meld_emotion/features/README.md) | 텍스트/오디오/비디오 × 임베딩/개념 추출기 | **새 특징 추출기** |
| [models/](src/meld_emotion/models/README.md) | 기초 학습기(Estimator) | **새 학습 알고리즘** |
| [fusion/](src/meld_emotion/fusion/README.md) | Early/Late fusion·결합기·모달리티 마스킹 | **새 융합/시나리오** |
| [evaluation/](src/meld_emotion/evaluation/README.md) | 지표·평가·강건성 | **새 지표/시나리오** |
| [explain/](src/meld_emotion/explain/README.md) | permutation·모달리티 ablation·반사실 | **새 설명기** |
| [pipeline/](src/meld_emotion/pipeline/README.md) | 특징 캐시·특징 파이프라인·러너·구성 루트 | 조립/오케스트레이션 |
| [reporting/](src/meld_emotion/reporting/README.md) | 콘솔/JSON/대시보드 리포터 | 새 출력 형식 |

## 실험 한 번의 흐름

```
DatasetSource → FeaturePipeline(추출기들) → FeatureBundle
            → Classifier(Early/Late fusion) → 학습
            → Evaluator + RobustnessEvaluator + Explainer 들
            → ExperimentResult → Reporter 들
```

설정(`ExperimentConfig`)이 위 모든 부품을 선택한다. 파이썬에서 직접 만들거나 YAML 로
기술할 수 있고, [pipeline/builder.py](src/meld_emotion/pipeline/builder.py) 가 설정을 구체
객체로 연결하는 **유일한 구성 루트**다(다른 모듈은 구체 구현을 import 하지 않는다).

세 층을 구분하면 어디를 고칠지 빨라진다:

- **무엇을 돌릴까(선언)** → `ExperimentConfig`/YAML. 최상위 변수 목록은
  [config/README.md](src/meld_emotion/config/README.md) 참고. 학습/평가 분할과 학습 시
  modality dropout 도 여기서 켠다(`train_split`/`eval_split`/`dropout`).
- **설정→객체 연결(조립)** → [pipeline/builder.py](src/meld_emotion/pipeline/builder.py).
- **어떤 순서로 실행할까(절차)** → [pipeline/runner.py](src/meld_emotion/pipeline/runner.py)
  `run()`. dev 기반 모델 선택·교차검증·다중 시드 같은 **절차 변경**은 여기서 한다.
- **여러 실험을 한 번에 비교** → `meld-emotion compare`(suite). 공유 `base` + 변형 목록을
  실행해 지표·강건성 비교표(콘솔+JSON)를 낸다. 일부 변형이 미구현 경계에 닿아도 나머지는
  계속 비교된다. 형식은 [pipeline/README.md](src/meld_emotion/pipeline/README.md) 참고.

## 구현 상태와 확장

- **무엇이 되어 있나**: `uv run meld-emotion status` 가 [core/status.py](src/meld_emotion/core/status.py)
  레지스트리에서 직접 읽어 REAL / PLACEHOLDER / UNIMPLEMENTED 를 출력한다. 손으로 관리하는
  목록이 아니므로 코드와 어긋나지 않는다.
- **무언가를 추가/교체하려면**: 해당 축의 패키지 README 의 "새 … 추가하기" 절을 따른다.
  공통 절차는 (1) Protocol 을 만족하는 클래스 작성 → (2) [config/schema.py](src/meld_emotion/config/schema.py)
  에 설정 dataclass 추가·등록 → (3) [pipeline/builder.py](src/meld_emotion/pipeline/builder.py)
  에 설정→구체 연결 한 줄 추가.
  - **예외(더 가벼움)**: 새 **지표**는 `METRIC_REGISTRY` 에 이름으로 등록만 하면 되고(빌더
    분기 불필요), 새 **강건성 시나리오**는 [fusion/masking.py](src/meld_emotion/fusion/masking.py)
    의 `SCENARIOS` 딕셔너리에 한 줄 추가하면 된다.
  - **예외(한 단계 더)**: 다른 설정을 품는 **중첩 설정**(예: `model.base`, `late.combiner`,
    `stacking.meta`)은 [config/loader.py](src/meld_emotion/config/loader.py) 에 재귀 복원도
    추가해야 YAML 에서 읽힌다.
