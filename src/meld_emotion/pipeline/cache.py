"""특징 캐시 구현.

특징 추출은 비용이 크므로(특히 12GB 비디오) 한 번 계산한 특징 행렬을 재사용한다.
:class:`InMemoryFeatureCache` / :class:`NullFeatureCache` 는 완전 구현이며,
:class:`DiskFeatureCache` 는 디스크 영속화가 구현되기 전까지 인메모리로 위임한다(임시).
"""

from __future__ import annotations

from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import note_placeholder_use, placeholder, real


@real
class InMemoryFeatureCache:
    """프로세스 메모리에 특징 행렬을 보관."""

    def __init__(self) -> None:
        self._store: dict[str, FeatureMatrix] = {}

    def get(self, key: str) -> FeatureMatrix | None:
        return self._store.get(key)

    def put(self, key: str, matrix: FeatureMatrix) -> None:
        self._store[key] = matrix


@real
class NullFeatureCache:
    """캐시를 사용하지 않음(항상 미스). 캐싱을 끄고 싶을 때."""

    def get(self, key: str) -> FeatureMatrix | None:
        return None

    def put(self, key: str, matrix: FeatureMatrix) -> None:
        return None


@placeholder("npz 등으로 디스크에 특징을 영속화해 실행 간 재사용해야 함 — 현재는 인메모리로 위임")
class DiskFeatureCache:
    """디스크 영속 캐시(임시: 인메모리)."""

    def __init__(self, path: str = ".feature_cache") -> None:
        self.path = path
        self._delegate = InMemoryFeatureCache()

    def get(self, key: str) -> FeatureMatrix | None:
        return self._delegate.get(key)

    def put(self, key: str, matrix: FeatureMatrix) -> None:
        note_placeholder_use(self)
        self._delegate.put(key, matrix)
