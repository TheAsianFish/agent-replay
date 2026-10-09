"""Judge providers: what turns a rubric and a transcript into verdicts.

- :class:`AnthropicJudge` (the default): ``claude-opus-5-5`` with structured
  outputs, effort ``low``; the rubric goes first in a cached system block.
- :class:`OllamaJudge`: a local model through ``ollama`` (``agent-replay[ollama]``),
  the offline fallback.
- :class:`FakeJudge`: deterministic and offline, for tests.

:func:`make_provider` picks one from a spec string (``REPLAY_JUDGE_PROVIDER``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import structlog
from pydantic import ValidationError

from agent_replay._loading import load_object
from agent_replay.cassette import Usage
from agent_replay.errors import AgentLoadError, ConfigError, JudgeError
from agent_replay.judge.rubric import Rubric
from agent_replay.judge.transcript import JudgeInput
from agent_replay.judge.verdict import (
    CriterionVerdict,
    JudgeOutput,
    JudgeResponse,
    system_prompt,
    validated,
)
from agent_replay.pricing import cost_usd

log = structlog.get_logger("agent_replay")

Timer = Callable[[], float]

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_EFFORT = "low"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
PROVIDERS = ("anthropic", "ollama", "fake")


class JudgeProvider(Protocol):
    """Anything that can grade a transcript against a rubric."""

    @property
    def judge_id(self) -> str:
        """What produces the verdicts (provider, model, settings); part of the cache key
        and matched against a calibration report."""
        ...

    def evaluate(self, rubric: Rubric, judge_input: JudgeInput, /) -> JudgeResponse: ...


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 3)


def resolve_effort(effort: str | None) -> str:
    value = effort if effort is not None else os.environ.get("REPLAY_JUDGE_EFFORT", DEFAULT_EFFORT)
    if value not in EFFORTS:
        source = "effort" if effort is not None else "REPLAY_JUDGE_EFFORT"
        raise ConfigError(f"{source}={value!r} is not an effort level; use {', '.join(EFFORTS)}")
    return value


def _model(model: str | None, default: str) -> str:
    value = model if model is not None else os.environ.get("REPLAY_JUDGE_MODEL") or default
    if not value.strip():
        raise ConfigError("the judge model name is empty; set REPLAY_JUDGE_MODEL or model=...")
    return value


def _parse(text: str, where: str) -> JudgeOutput:
    try:
        return JudgeOutput.model_validate_json(text)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "(root)"
        raise JudgeError(
            f"{where} returned an answer that does not match the verdict schema "
            f"({loc}: {first['msg']}); retry, or use a stronger judge model (REPLAY_JUDGE_MODEL)"
        ) from exc


# --- Anthropic ---------------------------------------------------------------------


def _no_credentials(exc: Exception) -> JudgeError:
    return JudgeError(
        f"the Anthropic judge has no credentials ({exc}); set ANTHROPIC_API_KEY, use "
        "REPLAY_JUDGE_PROVIDER=ollama, or judge from the cache only with REPLAY_JUDGE_OFFLINE=1"
    )


class AnthropicJudge:
    """Grade with Claude through structured outputs.

    The request puts the instructions and rubric in the system prompt with
    ``cache_control`` (stable across runs), the transcript in the user turn,
    the verdict schema in ``output_config.format`` and the effort in
    ``output_config.effort``. It never sends ``thinking``, ``budget_tokens`` or a
    forced ``tool_choice``, which Opus 5.5 rejects. ``client`` is any object with
    the ``anthropic.Anthropic`` ``messages.create`` interface; ``None`` creates
    ``anthropic.Anthropic()`` on first use.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int = 8000,
        timer: Timer = time.perf_counter,
    ) -> None:
        if max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {max_tokens}")
        self.model = _model(model, DEFAULT_ANTHROPIC_MODEL)
        self.effort = resolve_effort(effort)
        self.max_tokens = max_tokens
        self._client = client
        self._timer = timer

    @property
    def judge_id(self) -> str:
        return f"anthropic:{self.model}:effort={self.effort}"

    def request(self, rubric: Rubric, judge_input: JudgeInput) -> dict[str, Any]:
        """The keyword arguments sent to ``messages.create``.

        The verdict schema goes in ``output_config.format`` exactly as
        ``messages.parse(output_format=JudgeOutput)`` would send it; calling
        ``create`` keeps ``stop_reason`` inspectable before parsing (``parse``
        raises on a truncated or refused answer instead).
        """
        from agent_replay.record.anthropic import wire_output_format  # noqa: PLC0415

        kwargs, _ = wire_output_format(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt(rubric),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": judge_input.render()}],
                "output_format": JudgeOutput,
                "output_config": {"effort": self.effort},
            }
        )
        return kwargs

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # noqa: PLC0415 - only needed for live judging

            try:
                self._client = anthropic.Anthropic()
            except anthropic.AnthropicError as exc:
                raise _no_credentials(exc) from exc
        return self._client

    def evaluate(self, rubric: Rubric, judge_input: JudgeInput, /) -> JudgeResponse:
        import anthropic  # noqa: PLC0415

        client = self._get_client()
        kwargs = self.request(rubric, judge_input)
        start = self._timer()
        try:
            response = client.messages.create(**kwargs)
        except TypeError as exc:
            # The SDK only finds out there are no credentials when it builds the request.
            if "authentication" in str(exc):
                raise _no_credentials(exc) from exc
            raise
        except anthropic.AnthropicError as exc:
            raise JudgeError(
                f"the Anthropic judge call failed: {type(exc).__name__}: {exc}. Check "
                "ANTHROPIC_API_KEY, the model name (REPLAY_JUDGE_MODEL) and the network, or "
                "judge from the cache only with REPLAY_JUDGE_OFFLINE=1"
            ) from exc
        latency = _ms(start, self._timer())
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            raise JudgeError(
                "the judge model refused to grade this run (stop_reason refusal); review "
                "the transcript and rubric, or use another judge model"
            )
        if stop == "max_tokens":
            raise JudgeError(
                f"the judge ran out of output tokens (max_tokens={self.max_tokens}) before "
                "finishing its verdict; raise max_tokens or lower REPLAY_JUDGE_EFFORT"
            )
        if stop == "model_context_window_exceeded":
            raise JudgeError(
                "the transcript does not fit the judge model's context window; lower "
                "max_chars on the judge check"
            )
        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        output = _parse(text, f"the Anthropic judge ({self.model})")
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
        )
        cost = cost_usd(getattr(response, "model", None), usage)
        log.info(
            "judge.anthropic",
            model=self.model,
            effort=self.effort,
            latency_ms=latency,
            cost_usd=cost,
            cache_read_input_tokens=usage.cache_read_input_tokens,
        )
        return JudgeResponse(
            verdicts=validated(rubric, output), usage=usage, cost_usd=cost, latency_ms=latency
        )


# --- Ollama --------------------------------------------------------------------------


def _ollama_errors() -> tuple[type[Exception], ...]:
    try:
        import ollama  # noqa: PLC0415 - optional dependency
    except ImportError:
        return ()
    return (ollama.ResponseError, ollama.RequestError)


class OllamaJudge:
    """Grade with a local model through ``ollama`` (``pip install 'agent-replay[ollama]'``).

    The verdict schema goes in ``format`` and the temperature is 0. ``client``
    is an ``ollama.Client`` (or anything with its ``chat``); ``None`` creates one
    for ``host`` (default: ollama's own, ``OLLAMA_HOST`` or localhost) on first use.
    Local calls cost nothing.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str | None = None,
        host: str | None = None,
        timer: Timer = time.perf_counter,
    ) -> None:
        self.model = _model(model, DEFAULT_OLLAMA_MODEL)
        self.host = host
        self._client = client
        self._timer = timer

    @property
    def judge_id(self) -> str:
        return f"ollama:{self.model}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import ollama  # noqa: PLC0415 - optional dependency
            except ImportError as exc:
                raise JudgeError(
                    "the ollama judge needs the ollama package; install it with "
                    "pip install 'agent-replay[ollama]'"
                ) from exc
            self._client = ollama.Client(host=self.host)
        return self._client

    def evaluate(self, rubric: Rubric, judge_input: JudgeInput, /) -> JudgeResponse:
        client = self._get_client()
        start = self._timer()
        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt(rubric)},
                    {"role": "user", "content": judge_input.render()},
                ],
                format=JudgeOutput.model_json_schema(),
                options={"temperature": 0},
            )
        except (ConnectionError, OSError) as exc:
            raise JudgeError(
                f"cannot reach the Ollama server for the judge ({exc}); start it with "
                f"`ollama serve` and pull the model with `ollama pull {self.model}`"
            ) from exc
        except _ollama_errors() as exc:
            raise JudgeError(
                f"the Ollama judge call failed: {exc}; check the model name "
                f"(`ollama pull {self.model}`, REPLAY_JUDGE_MODEL)"
            ) from exc
        latency = _ms(start, self._timer())
        output = _parse(response.message.content or "", f"the Ollama judge ({self.model})")
        usage = Usage(
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        )
        log.info("judge.ollama", model=self.model, latency_ms=latency)
        return JudgeResponse(
            verdicts=validated(rubric, output), usage=usage, cost_usd=0.0, latency_ms=latency
        )


# --- fake -------------------------------------------------------------------------------

Decision = bool | tuple[bool, str]
Decide = Callable[[Rubric, JudgeInput], Mapping[str, Decision]]


def _pass_all(rubric: Rubric, _judge_input: JudgeInput) -> Mapping[str, Decision]:
    return dict.fromkeys(rubric.ids, True)


class FakeJudge:
    """A deterministic offline judge for tests (yours and ours).

    ``decide(rubric, judge_input)`` returns ``{criterion_id: passed}`` or
    ``{criterion_id: (passed, reasoning)}``; the default passes every criterion.
    The answer is validated like a real judge's, so leaving a criterion out is
    a :class:`JudgeError`. ``calls`` counts evaluations, to assert on caching.
    """

    def __init__(self, decide: Decide | None = None, *, name: str = "default") -> None:
        self.decide = decide or _pass_all
        self.name = name
        self.calls = 0

    @property
    def judge_id(self) -> str:
        return f"fake:{self.name}"

    def evaluate(self, rubric: Rubric, judge_input: JudgeInput, /) -> JudgeResponse:
        self.calls += 1
        verdicts = []
        for criterion_id, decision in self.decide(rubric, judge_input).items():
            passed, reasoning = (
                decision if isinstance(decision, tuple) else (decision, "decided by the fake judge")
            )
            verdicts.append(
                CriterionVerdict(
                    id=criterion_id, reasoning=reasoning, verdict="pass" if passed else "fail"
                )
            )
        output = JudgeOutput(criteria=verdicts)
        return JudgeResponse(
            verdicts=validated(rubric, output), usage=None, cost_usd=0.0, latency_ms=0.0
        )


# --- selection -------------------------------------------------------------------------------


def make_provider(
    spec: str | None = None, *, model: str | None = None, effort: str | None = None
) -> JudgeProvider:
    """A provider from ``"anthropic"``, ``"ollama"``, ``"fake"`` or ``"module:factory"``.

    ``spec`` defaults to ``REPLAY_JUDGE_PROVIDER``, then ``anthropic``. ``model``
    applies to anthropic and ollama, ``effort`` to anthropic only. A factory is
    called with no arguments and must return a :class:`JudgeProvider`.
    """
    source = "provider" if spec is not None else "REPLAY_JUDGE_PROVIDER"
    value = spec if spec is not None else os.environ.get("REPLAY_JUDGE_PROVIDER") or "anthropic"
    if value == "anthropic":
        return AnthropicJudge(model=model, effort=effort)
    if effort is not None:
        raise ConfigError(f"effort applies to the anthropic judge only, not {value!r}")
    if value == "ollama":
        return OllamaJudge(model=model)
    if model is not None:
        raise ConfigError(f"model applies to the anthropic and ollama judges, not {value!r}")
    if value == "fake":
        return FakeJudge()
    if ":" not in value:
        raise ConfigError(
            f"{source}={value!r} is not a judge provider; use {', '.join(PROVIDERS)} or "
            "'module:factory' returning a judge provider"
        )
    try:
        factory = load_object(value)
    except AgentLoadError as exc:
        raise ConfigError(f"cannot load judge provider {value!r}: {exc}") from exc
    if not callable(factory):
        raise ConfigError(f"judge provider {value!r} is not callable; point it at a factory")
    try:
        provider = factory()
    except Exception as exc:
        raise ConfigError(
            f"judge provider factory {value!r} raised {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(getattr(provider, "judge_id", None), str) or not callable(
        getattr(provider, "evaluate", None)
    ):
        raise ConfigError(
            f"judge provider factory {value!r} returned {type(provider).__name__}, which has no "
            "judge_id string and evaluate(rubric, judge_input) method"
        )
    result: JudgeProvider = provider
    return result
