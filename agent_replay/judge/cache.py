"""A content-addressed cache of judge verdicts on disk.

The key is a hash of everything that decides a verdict: the judge (provider,
model, settings), the rubric fingerprint, the transcript fingerprint and the
prompt version. A ``full``-mode replay produces the same transcript every time,
so after the first run it is judged for free and deterministically. Commit the
cache directory and CI can judge offline (``REPLAY_JUDGE_OFFLINE=1``): a miss
then raises :class:`JudgeCacheMiss` instead of calling the provider, and
``replay ci`` exits 2 so a stale committed cache cannot pass unnoticed.

Layout: ``<directory>/<key[:2]>/<key>.json``, one verdict per file, written
atomically. A corrupt file counts as a miss.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from agent_replay._json import canonical_json
from agent_replay.cassette import Usage
from agent_replay.errors import ConfigError, JudgeCacheMiss, JudgeError
from agent_replay.judge.providers import JudgeProvider
from agent_replay.judge.rubric import Rubric
from agent_replay.judge.transcript import JudgeInput
from agent_replay.judge.verdict import (
    PROMPT_VERSION,
    CriterionVerdict,
    JudgeOutput,
    JudgeResponse,
    validated,
)

log = structlog.get_logger("agent_replay")

DEFAULT_DIRECTORY = ".replay/judge-cache"
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


def env_flag(name: str) -> bool:
    """A boolean environment variable; anything unrecognised is a :class:`ConfigError`."""
    raw = os.environ.get(name, "")
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean; use 1/true or 0/false")


def _file_mode() -> int:
    """``0o666`` less the process umask: what ``open()`` would have created."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def cache_key(judge_id: str, rubric: Rubric, judge_input: JudgeInput) -> str:
    payload = canonical_json(
        {
            "judge_id": judge_id,
            "rubric": rubric.fingerprint,
            "input": judge_input.fingerprint,
            "prompt_version": PROMPT_VERSION,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CacheEntry(BaseModel):
    """One cached verdict, with what it originally cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    judge_id: str
    rubric: str
    rubric_fingerprint: str
    input_fingerprint: str
    prompt_version: str
    verdicts: tuple[CriterionVerdict, ...]
    usage: Usage | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    created_at: datetime

    def response(self) -> JudgeResponse:
        return JudgeResponse(
            verdicts=self.verdicts,
            usage=self.usage,
            cost_usd=self.cost_usd,
            latency_ms=self.latency_ms,
        )


class JudgeCache:
    """Verdicts on disk under ``directory`` (default ``REPLAY_JUDGE_CACHE``, then
    ``.replay/judge-cache``). With ``offline`` (default ``REPLAY_JUDGE_OFFLINE``) a
    miss raises :class:`JudgeError` instead of calling the provider."""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        offline: bool | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.directory = Path(
            directory
            if directory is not None
            else os.environ.get("REPLAY_JUDGE_CACHE") or DEFAULT_DIRECTORY
        )
        self.offline = offline if offline is not None else env_flag("REPLAY_JUDGE_OFFLINE")
        self._clock = clock

    def path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        path = self.path(key)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("judge.cache_unreadable", path=str(path), error=str(exc))
            return None
        try:
            entry = CacheEntry.model_validate_json(text)
        except ValidationError as exc:
            log.warning("judge.cache_corrupt", path=str(path), error=str(exc.errors()[0]["msg"]))
            return None
        if entry.key != key:
            log.warning("judge.cache_corrupt", path=str(path), error="key does not match the file")
            return None
        return entry

    def _matching(
        self, key: str, judge_id: str, rubric: Rubric, judge_input: JudgeInput
    ) -> CacheEntry | None:
        """The entry for ``key`` if its own fields describe this judge, rubric, transcript and
        prompt version; an entry copied under another key counts as corrupt."""
        entry = self.get(key)
        if entry is None:
            return None
        expected = {
            "judge_id": judge_id,
            "rubric_fingerprint": rubric.fingerprint,
            "input_fingerprint": judge_input.fingerprint,
            "prompt_version": PROMPT_VERSION,
        }
        wrong = [f for f, v in expected.items() if getattr(entry, f) != v]
        if wrong:
            log.warning(
                "judge.cache_corrupt",
                path=str(self.path(key)),
                error=f"{', '.join(wrong)} do(es) not match the key",
            )
            return None
        return entry

    def put(
        self,
        key: str,
        judge_id: str,
        rubric: Rubric,
        judge_input: JudgeInput,
        response: JudgeResponse,
    ) -> CacheEntry:
        entry = CacheEntry(
            key=key,
            judge_id=judge_id,
            rubric=rubric.name,
            rubric_fingerprint=rubric.fingerprint,
            input_fingerprint=judge_input.fingerprint,
            prompt_version=PROMPT_VERSION,
            verdicts=response.verdicts,
            usage=response.usage,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            created_at=self._clock(),
        )
        path = self.path(key)
        tmp_name: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            os.chmod(fd, _file_mode())  # mkstemp creates 0600; a shared cache should be readable
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                tmp.write(entry.model_dump_json(indent=2) + "\n")
            os.replace(tmp_name, path)
        except OSError as exc:
            # A verdict that cannot be cached is still a verdict.
            log.warning("judge.cache_write_failed", path=str(path), error=str(exc))
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)
        return entry

    def verdict(
        self, provider: JudgeProvider, rubric: Rubric, judge_input: JudgeInput
    ) -> tuple[JudgeResponse, bool]:
        """The cached verdict, else the provider's (then cached). Returns ``(response, hit)``."""
        key = cache_key(provider.judge_id, rubric, judge_input)
        entry = self._matching(key, provider.judge_id, rubric, judge_input)
        if entry is not None:
            try:
                verdicts = validated(rubric, JudgeOutput(criteria=list(entry.verdicts)))
            except JudgeError as exc:  # a hand-edited entry
                log.warning("judge.cache_corrupt", path=str(self.path(key)), error=str(exc))
            else:
                return entry.response().model_copy(update={"verdicts": verdicts}), True
        if self.offline:
            raise JudgeCacheMiss(
                f"this run's verdict is not in the judge cache {str(self.directory)!r} (judge "
                f"{provider.judge_id}, rubric {rubric.name!r}) and offline mode is on; run once "
                "with the provider available, or commit the cache directory"
            )
        response = provider.evaluate(rubric, judge_input)
        self.put(key, provider.judge_id, rubric, judge_input, response)
        return response, False
