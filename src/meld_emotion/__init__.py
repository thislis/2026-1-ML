"""MELD 멀티모달 감정 인식 파이프라인 (CSE363 기말 프로젝트).

이 패키지는 텍스트/오디오/비디오 모달리티로부터 특징을 추출하고, Early/Late fusion으로
결합하여 발화 단위 감정을 분류하는 모듈형 파이프라인을 제공한다. 각 단계는
`meld_emotion.core.protocols` 의 Protocol로 정의된 계약을 따르며, 서로 느슨하게 결합되어
있어 데이터셋/특징/모델/융합 전략을 독립적으로 교체할 수 있다.

전체 구조와 사용법은 최상위 ``README.md`` 를 참고하라.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

__all__ = ["__version__"]

logging.getLogger(__name__).addHandler(logging.NullHandler())
