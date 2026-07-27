"""Repository path configuration helpers.

Data JSON files store logical paths relative to ``dermogpt-harness.yaml`` roots.
Runtime code resolves those paths at the boundary where files are opened.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - dependency smoke catches this.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARNESS_CONFIG = PROJECT_ROOT / "dermogpt-harness.yaml"


@lru_cache(maxsize=4)
def load_harness_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = Path(
        config_path
        or os.environ.get("DERMOGPT_HARNESS_CONFIG")
        or DEFAULT_HARNESS_CONFIG
    )
    if not path.exists():
        raise FileNotFoundError(f"DermoGPT harness config not found: {path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read dermogpt-harness.yaml")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_paths_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Backward-compatible alias for callers that only need path sections."""
    return load_harness_config(config_path)


def _config_section(name: str) -> dict[str, Any]:
    return load_harness_config().get(name, {}) or {}


def _first_asset_root(kind: str) -> str | None:
    roots = (
        _config_section("assets_root_path")
        .get(kind, {})
        .get("roots", [])
    )
    return str(roots[0]) if roots else None


def root_path(name: str) -> Path:
    canonical = {
        "project_root": _config_section("project").get("root"),
        "dataset_root": _first_asset_root("datasets"),
        "model_root": _first_asset_root("models"),
        "temp_root": _config_section("paths").get("temp_root"),
    }
    value = (
        canonical.get(name)
        or _config_section("roots").get(name)
        or _config_section("paths").get(name)
    )
    if not value:
        raise KeyError(f"dermogpt-harness.yaml missing canonical root for {name}")
    return Path(str(value)).expanduser()


def dataset_root() -> Path:
    return root_path("dataset_root")


def model_root() -> Path:
    return root_path("model_root")


def project_root() -> Path:
    return root_path("project_root")


def _resolve_relative(value: str | os.PathLike[str], root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return root / path


def resolve_dataset_path(value: str | os.PathLike[str]) -> Path:
    return _resolve_relative(value, dataset_root())


def resolve_model_path(value: str | os.PathLike[str]) -> Path:
    return _resolve_relative(value, model_root())


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    return _resolve_relative(value, project_root())


def configured_model_path(name: str) -> Path:
    value = _config_section("models").get(name) or _config_section("assets").get("models", {}).get(name)
    if not value:
        raise KeyError(f"dermogpt-harness.yaml missing models.{name}")
    return resolve_model_path(value)


def configured_dataset_path(name: str) -> Path:
    value = _config_section("datasets").get(name) or _config_section("assets").get("datasets", {}).get(name)
    if not value:
        raise KeyError(f"dermogpt-harness.yaml missing datasets.{name}")
    return resolve_dataset_path(value)


def _strip_root(path_text: str, roots: list[Path]) -> str:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        return path_text
    for root in roots:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return path_text


def to_logical_path(value: Any) -> Any:
    """Return a stable sync-friendly path string when ``value`` is path-like.

    Absolute paths under configured roots become POSIX relative paths. ``zip://``
    paths preserve their ``!inner/path`` suffix while normalizing the archive
    path itself.
    """
    if not isinstance(value, str) or not value:
        return value
    roots = [dataset_root(), model_root(), project_root()]
    if value.startswith("zip://"):
        body = value[len("zip://") :]
        archive, sep, inner = body.partition("!")
        archive = _strip_root(archive, roots)
        return f"zip://{archive}{sep}{inner}" if sep else f"zip://{archive}"
    return _strip_root(value, roots)
