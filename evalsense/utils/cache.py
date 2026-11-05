"""Stable cache identities for declared configuration, not arbitrary Python code."""

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class CacheIdentityError(ValueError):
    """A component needs an explicit version before its output can be reused."""


@lru_cache(maxsize=1)
def _implementation_versions() -> dict[str, str]:
    return {name: package_version(name) for name in ("evalsense", "inspect-ai")}


def _normalise(value: Any, version: str | None) -> Any:
    if value is None or type(value) in (str, bool, int):
        return [type(value).__name__, value]
    if type(value) is float:
        if not math.isfinite(value):
            raise CacheIdentityError("Cache configuration must contain finite numbers.")
        return ["float", value.hex()]
    if isinstance(value, Mapping):
        items = [
            (_normalise(k, version), _normalise(v, version)) for k, v in value.items()
        ]
        return ["dict", sorted(items, key=lambda item: json.dumps(item[0]))]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_normalise(v, version) for v in value]]
    if isinstance(value, Sequence) and type(value).__module__.startswith("inspect_ai."):
        # Inspect chains expose their solvers as a sequence; inspect each solver
        # instead of trusting a serialized name for a nested custom callable.
        return ["sequence", _name(type(value)), [_normalise(v, version) for v in value]]
    if isinstance(value, Path):
        return ["path", str(value)]
    if isinstance(value, BaseModel):
        return ["model", _name(type(value)), _normalise(value.model_dump(), version)]
    if is_dataclass(value) and not isinstance(value, type):
        return [
            "dataclass",
            _name(type(value)),
            _normalise(
                {field.name: getattr(value, field.name) for field in fields(value)},
                version,
            ),
        ]
    if callable(value):
        name = _name(value)
        module = getattr(value, "__module__", "")
        if version is None and not module.startswith(("inspect_ai.", "evalsense.")):
            raise CacheIdentityError(
                f"Custom component {name!r} needs an explicit cache version. "
                "Set its version and bump it when code or external inputs change."
            )
        try:
            # Inspect's decorators attach registry metadata and construction
            # parameters. The public registry helpers are absent in older
            # supported releases, so keep attribute access in this one adapter.
            info = getattr(value, "__registry_info__", None)
            params = getattr(value, "__registry_params__", None)
            if (
                info is None
                or not isinstance(getattr(info, "name", None), str)
                or not isinstance(getattr(info, "type", None), str)
                or not isinstance(getattr(info, "metadata", None), dict)
                or not isinstance(params, dict)
            ):
                raise ValueError("No construction parameters")
        except ValueError:
            if version is None and not module.startswith("evalsense."):
                raise CacheIdentityError(
                    f"Component {name!r} has no declared parameters; set a version."
                ) from None
            return ["callable", name, version]
        return [
            "registered",
            info.type,
            info.name,
            name,
            version,
            _normalise(params, version),
            _normalise(info.metadata, version),
        ]
    if version is not None:
        # The supplied version is the caller's contract for an opaque object's
        # behavior. Never persist repr(), addresses, or mutable implementation state.
        return ["versioned", _name(type(value)), version]
    raise CacheIdentityError(
        f"Cannot identify {_name(type(value))!r}; supply an explicit component version."
    )


def _name(value: Any) -> str:
    return f"{getattr(value, '__module__', type(value).__module__)}." + str(
        getattr(value, "__qualname__", type(value).__qualname__)
    )


def fingerprint(value: Any, *, version: str | None = None) -> str:
    """Hash declared parameters and an optional version for custom behavior.

    A version covers code, closure state, files, and other dependencies not declared
    as parameters. It must be changed by the caller when those inputs change.
    """
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise CacheIdentityError("A cache version must be a non-empty string.")
    payload = [
        "evalsense-cache-v1",
        _implementation_versions(),
        version,
        _normalise(value, version),
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
