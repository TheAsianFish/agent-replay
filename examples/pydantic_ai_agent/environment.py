"""What a local-model measurement ran on: the CPU, the Ollama server and the model's digest.

A local model's latency and its sampling depend on all three, so every live
recording and flake report of the travel desk carries them.
"""

from __future__ import annotations

import json
import os
import platform
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from examples.pydantic_ai_agent.agent import MODEL, ollama_url


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _get(path: str) -> Any:
    with urllib.request.urlopen(f"{ollama_url()}{path}", timeout=5) as response:  # noqa: S310 - local server URL from config
        return json.loads(response.read())


def ollama_model(model: str = MODEL) -> dict[str, str] | None:
    """The Ollama server's version and ``model``'s digest, or ``None`` if either is missing."""
    try:
        version = _get("/api/version")["version"]
        tags = _get("/api/tags")["models"]
    except (OSError, urllib.error.URLError, KeyError, ValueError):
        return None
    for entry in tags:
        if entry.get("name") == model or entry.get("model") == model:
            return {"ollama": str(version), "model": model, "digest": str(entry.get("digest"))}
    return None


def describe(model: str = MODEL) -> dict[str, Any]:
    """Machine and server facts for a report: CPU model, logical CPUs, Ollama and the model."""
    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        **(ollama_model(model) or {"model": model}),
    }
