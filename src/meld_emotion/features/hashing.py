"""프로세스 간 안정적인 해시 유틸 (해싱 특징의 재현성 보장).

파이썬의 내장 ``hash`` 는 문자열에 대해 프로세스마다 달라지므로(PYTHONHASHSEED) 특징 캐시
재사용 시 결과가 어긋난다. 여기서는 ``zlib.crc32`` 로 결정적 해시를 제공한다.
"""

from __future__ import annotations

import zlib


def stable_hash(text: str) -> int:
    """문자열의 결정적 비음수 해시."""

    return zlib.crc32(text.encode("utf-8"))
