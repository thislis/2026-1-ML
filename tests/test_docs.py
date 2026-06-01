"""문서 거버넌스: README 존재와 상태 태그 일관성(소스↔문서 드리프트 방지)."""

from __future__ import annotations

from pathlib import Path

# builder 를 import 하면 모든 구체 컴포넌트가 로드되어 상태 레지스트리가 채워진다.
from meld_emotion.core.status import ComponentStatus, iter_status
from meld_emotion.pipeline import builder  # noqa: F401

_SRC = Path(__file__).resolve().parents[1] / "src" / "meld_emotion"

REQUIRED_PACKAGES = (
    "core",
    "config",
    "data",
    "features",
    "models",
    "fusion",
    "evaluation",
    "explain",
    "pipeline",
    "reporting",
)


def test_root_readme_exists() -> None:
    root = Path(__file__).resolve().parents[1] / "README.md"
    assert root.exists() and root.read_text(encoding="utf-8").strip()


def test_each_package_has_readme() -> None:
    missing = [pkg for pkg in REQUIRED_PACKAGES if not (_SRC / pkg / "README.md").exists()]
    assert not missing, f"README.md 누락 패키지: {missing}"


def test_status_reasons_present() -> None:
    for record in iter_status():
        if record.status in (ComponentStatus.PLACEHOLDER, ComponentStatus.UNIMPLEMENTED):
            assert record.reason.strip(), f"{record.qualname} 에 사유(reason)가 없습니다"


def test_registry_has_expected_components() -> None:
    statuses = [r.status for r in iter_status()]
    assert statuses.count(ComponentStatus.REAL) >= 20
    assert ComponentStatus.PLACEHOLDER in statuses
    assert ComponentStatus.UNIMPLEMENTED in statuses
