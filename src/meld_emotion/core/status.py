"""구현 상태(status) 마커와 레지스트리.

각 컴포넌트가 "완전 구현(REAL) / 임시 기본동작(PLACEHOLDER) / 미구현(UNIMPLEMENTED)" 중
무엇인지를 **코드에 직접** 표시하고, 이를 한 곳(레지스트리)에 모은다. 덕분에 진행 상황을
손으로 관리하는 문서 없이 ``meld-emotion status`` 명령으로 항상 최신 상태를 출력할 수 있다.

설계 원칙(plan 참고):
- ``@placeholder``: 프로토콜을 만족하는 **안전한 기본 동작**(numpy만 사용, 결정적)을 제공해
  하위 컴포넌트 테스트가 막히지 않게 한다. 기본 출력 생성 시 ``note_placeholder_use`` 를
  호출해 (클래스당 한 번) ``RuntimeWarning`` 을 남긴다.
- ``@unimplemented``: 가짜 출력이 위험하거나 오해를 부르는 경우. 메서드에서
  ``raise_unimplemented(self)`` 로 명시적으로 ``NotImplementedError`` 를 발생시킨다.
- 태그가 없으면 기본값은 REAL 로 간주한다(명시하고 싶으면 ``@real`` 사용).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn, TypeVar

_T = TypeVar("_T", bound=type)


class ComponentStatus(StrEnum):
    """컴포넌트 구현 상태 3단계."""

    REAL = "real"
    PLACEHOLDER = "placeholder"
    UNIMPLEMENTED = "unimplemented"


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    """레지스트리에 저장되는 한 컴포넌트의 상태 기록."""

    qualname: str
    status: ComponentStatus
    reason: str


_REGISTRY: dict[type, ComponentRecord] = {}
_WARNED: set[type] = set()


def _register(cls: type, status: ComponentStatus, reason: str) -> None:
    qualname = f"{cls.__module__}.{cls.__qualname__}"
    _REGISTRY[cls] = ComponentRecord(qualname=qualname, status=status, reason=reason)


def real(cls: _T) -> _T:
    """완전 구현된 컴포넌트임을 명시적으로 표시한다(선택 사항)."""

    _register(cls, ComponentStatus.REAL, "")
    return cls


def placeholder(reason: str) -> Callable[[_T], _T]:
    """임시 기본 동작 컴포넌트로 표시하는 데코레이터.

    Args:
        reason: 실제 구현이 무엇이어야 하는지에 대한 한 줄 설명. ``status`` 출력과
            할 일 목록에 그대로 사용된다.
    """

    def deco(cls: _T) -> _T:
        _register(cls, ComponentStatus.PLACEHOLDER, reason)
        return cls

    return deco


def unimplemented(reason: str) -> Callable[[_T], _T]:
    """미구현 컴포넌트로 표시하는 데코레이터.

    Args:
        reason: 어떤 구현이 필요한지에 대한 설명. ``raise_unimplemented`` 가 던지는
            예외 메시지로 재사용된다.
    """

    def deco(cls: _T) -> _T:
        _register(cls, ComponentStatus.UNIMPLEMENTED, reason)
        return cls

    return deco


def record_of(cls: type) -> ComponentRecord | None:
    """해당 클래스의 상태 기록을 반환(태그가 없으면 ``None``)."""

    return _REGISTRY.get(cls)


def status_of(cls: type) -> ComponentStatus:
    """클래스의 상태를 반환. 태그가 없으면 REAL 로 간주한다."""

    record = _REGISTRY.get(cls)
    return record.status if record is not None else ComponentStatus.REAL


def reason_of(cls: type) -> str:
    record = _REGISTRY.get(cls)
    return record.reason if record is not None else ""


def iter_status() -> Iterator[ComponentRecord]:
    """등록된 모든 상태 기록을 (qualname 기준 정렬) 순회한다."""

    yield from sorted(_REGISTRY.values(), key=lambda r: r.qualname)


def note_placeholder_use(obj: object) -> None:
    """PLACEHOLDER 컴포넌트가 가짜 기본 출력을 생성할 때 호출. 클래스당 한 번 경고."""

    cls = type(obj)
    if cls in _WARNED:
        return
    _WARNED.add(cls)
    reason = reason_of(cls) or "임시 기본 동작이 사용되었습니다."
    warnings.warn(
        f"[placeholder] {cls.__module__}.{cls.__qualname__}: {reason}",
        RuntimeWarning,
        stacklevel=2,
    )


def raise_unimplemented(obj: object) -> NoReturn:
    """UNIMPLEMENTED 컴포넌트의 메서드에서 호출해 명시적 예외를 던진다."""

    cls = type(obj)
    reason = reason_of(cls) or "아직 구현되지 않았습니다."
    raise NotImplementedError(f"{cls.__module__}.{cls.__qualname__}: {reason}")
