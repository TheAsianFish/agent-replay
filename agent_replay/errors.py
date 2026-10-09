"""Exceptions raised by agent-replay.

Every message names the fix, because these surface in CI logs where the reader
has no debugger.
"""

from __future__ import annotations

import sys


class ReplayError(Exception):
    """Base class for every error agent-replay raises on purpose."""


class CassetteError(ReplayError):
    """A cassette is missing, malformed, or written by an incompatible version."""


class AgentLoadError(ReplayError):
    """An ``module:attribute`` reference could not be imported."""


class UnrecordedCallError(ReplayError):
    """During replay the agent made a call that the cassette has no envelope for."""

    def __init__(
        self, kind: str, name: str, request_preview: str, *, hint: str | None = None
    ) -> None:
        self.kind = kind
        self.name = name
        fix = hint or (
            "If the change is intended, re-record the cassette with `replay record`, or "
            "replay with --policy fallback (perform it for real) or --policy synth (stub it)."
        )
        super().__init__(
            f"the agent made a {kind} call {name!r} that the cassette never recorded "
            f"(or made it more times than recorded); request: {request_preview}. {fix}"
        )


class RecordedException(Exception):
    """Raised during replay where the original call raised.

    When the original class is already imported (builtins, the SDK, the agent's
    own modules), replay raises an instance of a subclass of *both* this class
    and the original, so ``except KeyError:`` handles it exactly as it did during
    recording. ``str()`` is always the original message.
    """

    def __init__(self, type_name: str, message: str, module: str | None = None) -> None:
        self.type_name = type_name
        self.module = module
        self.message = message
        # Not super(): in a replay hybrid the next class is the original exception,
        # whose __init__ may require arguments we do not have.
        Exception.__init__(self, message)

    def __str__(self) -> str:
        return self.message


def _resolve_class(module: str | None, qualname: str) -> type[Exception] | None:
    # Only look at modules that are already imported: replay must not import
    # code (and run its side effects) just because a cassette names it.
    obj: object = sys.modules.get(module or "builtins")
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
    if isinstance(obj, type) and issubclass(obj, Exception):
        return obj
    return None


def rebuild_exception(type_name: str, module: str | None, message: str) -> RecordedException:
    """An exception to raise on replay for a recorded failure (see :class:`RecordedException`)."""
    original = _resolve_class(module, type_name)
    if original is None or issubclass(original, RecordedException):
        return RecordedException(type_name, message, module)
    try:
        hybrid = type(original.__name__, (RecordedException, original), {"__module__": __name__})
        # Bypass the original __init__, whose signature we cannot know (SDK API
        # errors require a response object, for example).
        exc: RecordedException = original.__new__(hybrid)
        RecordedException.__init__(exc, type_name, message, module)
    except TypeError:  # incompatible base layout
        return RecordedException(type_name, message, module)
    return exc


class StoreError(ReplayError):
    """The run store could not be opened, or a stored run could not be found."""


class ConfigError(ReplayError):
    """A check or suite configuration table is invalid."""
