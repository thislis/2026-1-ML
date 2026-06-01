"""이름 → 팩토리 매핑을 제공하는 일반 레지스트리.

하이브리드 설정 방식에서 **YAML 경계**에만 쓰인다. 즉 ``type: "tfidf"`` 같은 문자열을
대응하는 설정 dataclass 로 복원할 때 사용한다. 파이썬 코드에서 설정을 직접 생성할 때는
레지스트리를 거치지 않으므로 정적 분석이 온전히 유지된다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """문자열 이름으로 팩토리(주로 dataclass 생성자)를 등록/조회한다."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    @property
    def kind(self) -> str:
        return self._kind

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """데코레이터 형태 등록."""

        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            self.add(name, factory)
            return factory

        return deco

    def add(self, name: str, factory: Callable[..., T]) -> None:
        if name in self._factories:
            raise ValueError(f"{self._kind} 레지스트리에 이미 '{name}' 이(가) 있습니다")
        self._factories[name] = factory

    def get(self, name: str) -> Callable[..., T]:
        try:
            return self._factories[name]
        except KeyError:
            raise KeyError(
                f"{self._kind} 레지스트리에 '{name}' 이(가) 없습니다. "
                f"사용 가능: {sorted(self._factories)}"
            ) from None

    def create(self, name: str, /, **kwargs: object) -> T:
        return self.get(name)(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        return name in self._factories
