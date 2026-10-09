"""Import ``module:attribute`` references (agents, clients, rules, schema models)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from agent_replay.errors import AgentLoadError


def load_object(ref: str) -> object:
    """Import ``"package.module:attribute"``; the working directory is importable."""
    module_name, sep, attr = ref.partition(":")
    if not sep or not module_name or not attr:
        raise AgentLoadError(f"{ref!r} is not a reference; use the form 'package.module:function'")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AgentLoadError(
            f"cannot import {module_name!r} for {ref!r}: {exc}. Run from the project root "
            "or install the package that contains the agent."
        ) from exc
    except Exception as exc:  # the module itself is broken: SyntaxError, error at import
        raise AgentLoadError(
            f"importing {module_name!r} for {ref!r} failed: {type(exc).__name__}: {exc}"
        ) from exc
    obj: object = module
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise AgentLoadError(f"{module_name!r} has no attribute {attr!r}") from exc
    return obj
