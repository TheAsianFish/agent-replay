"""``rule``: an arbitrary Python predicate over the run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_replay.checks.base import CheckResult

if TYPE_CHECKING:
    from agent_replay.runner import RunResult

RuleFn = Callable[..., "bool | tuple[bool, str] | CheckResult"]


@dataclass(frozen=True)
class Rule:
    """Call ``fn(run, **kwargs)``.

    ``fn`` may return a ``bool``, a ``(bool, message)`` pair or a full
    :class:`CheckResult`. An exception from ``fn`` fails the check (with the
    exception type and message) instead of crashing the suite.
    """

    fn: RuleFn = field(compare=False)
    rule_name: str
    kwargs: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def name(self) -> str:
        return self.rule_name

    def __call__(self, run: RunResult, /) -> CheckResult:
        try:
            out = self.fn(run, **self.kwargs)
        except Exception as exc:
            return CheckResult(
                check=self.name,
                passed=False,
                message=f"rule raised {type(exc).__name__}: {exc}",
                details={"exception": type(exc).__name__},
            )
        if isinstance(out, CheckResult):
            return out
        if isinstance(out, bool):
            return CheckResult(
                check=self.name, passed=out, message="rule passed" if out else "rule returned False"
            )
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
            return CheckResult(check=self.name, passed=out[0], message=str(out[1]))
        return CheckResult(
            check=self.name,
            passed=False,
            message=(
                f"rule returned {out!r}; return a bool, a (bool, message) pair or a CheckResult"
            ),
        )


def rule(fn: RuleFn, *, name: str | None = None, **kwargs: Any) -> Rule:
    """Build a ``rule`` check from a predicate. See :class:`Rule`."""
    if not callable(fn):
        raise TypeError(f"rule needs a callable, got {fn!r}")
    label = name or f"rule:{getattr(fn, '__name__', type(fn).__name__)}"
    return Rule(fn=fn, rule_name=label, kwargs=dict(kwargs))
