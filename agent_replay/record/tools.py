"""The ``@tool`` decorator: record and replay plain Python tool functions."""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Awaitable, Callable
from typing import Any, overload

from anthropic.types import ToolParam
from pydantic import (
    JsonValue,
    PydanticSchemaGenerationError,
    PydanticUserError,
    TypeAdapter,
    create_model,
)

from agent_replay._json import fingerprint, to_jsonable
from agent_replay.cassette import Kind
from agent_replay.errors import ReplayError
from agent_replay.session import Call, current_session
from agent_replay.synth import SynthError, stub_from_schema


class Tool[**P, R]:
    """A tool function that records its calls and can be served from a cassette.

    Calling it behaves exactly like calling the function when no session is
    active. Arguments are recorded by parameter name (defaults applied), and on
    replay the recorded JSON result is validated back into the function's
    annotated return type, so ``-> Order`` returns an ``Order`` both times.

    An ``async def`` tool stays a coroutine function: calling it returns an
    awaitable, which inside a session goes through ``session.intercept_async``
    when awaited, so tools run with ``asyncio.gather`` are recorded with their
    nesting (each task copies the context holding the parent span).
    """

    def __init__(
        self, fn: Callable[P, R], *, name: str | None = None, description: str | None = None
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or inspect.getdoc(fn) or ""
        self._signature = inspect.signature(fn)
        for param in self._signature.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(f"tool {self.name!r}: *args and **kwargs are not supported")
        functools.update_wrapper(self, fn)
        self.__name__ = self.name
        self.is_async = inspect.iscoroutinefunction(fn)
        if self.is_async:
            # So inspect.iscoroutinefunction(tool) is true, as it is for the function.
            inspect.markcoroutinefunction(self)

    # Type hints are resolved lazily: decorating must never break an import
    # (forward references, annotations pydantic cannot schematise), because
    # outside a session the tool is just the function.
    @functools.cached_property
    def _hints(self) -> dict[str, Any]:
        try:
            return typing.get_type_hints(self.fn, include_extras=True)
        except NameError as exc:
            raise ReplayError(f"tool {self.name!r}: cannot resolve type hints ({exc})") from exc

    @functools.cached_property
    def _returns(self) -> TypeAdapter[Any] | None:
        hint = self._hints.get("return")
        if hint is None or hint is type(None):
            return None
        try:
            return TypeAdapter(hint)
        except PydanticSchemaGenerationError as exc:
            raise ReplayError(
                f"tool {self.name!r}: return type {hint!r} cannot be rebuilt from JSON on "
                "replay; return JSON-compatible data or a pydantic model"
            ) from exc

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        session = current_session()
        if session is None:
            return self.fn(*args, **kwargs)
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        try:
            request = to_jsonable(dict(bound.arguments))
            decode = self._decoder()
        except (TypeError, ReplayError) as exc:
            raise session.fail(
                exc
                if isinstance(exc, ReplayError)
                else ReplayError(
                    f"tool {self.name!r}: arguments cannot be recorded as JSON ({exc}); "
                    "pass JSON-compatible values"
                )
            ) from exc
        if self.is_async:
            # The awaited value is what gets recorded; get_type_hints on an
            # ``async def`` already gives its awaited return type for decoding.
            acall: Call[Any] = Call(
                kind=Kind.TOOL,
                name=self.name,
                request=request,
                fingerprint=fingerprint(Kind.TOOL, self.name, request),
                ainvoke=lambda: typing.cast(Awaitable[Any], self.fn(*args, **kwargs)),
                encode=to_jsonable,
                decode=decode,
                synth=self._synth,
            )
            return typing.cast(R, session.intercept_async(acall))
        call: Call[R] = Call(
            kind=Kind.TOOL,
            name=self.name,
            request=request,
            fingerprint=fingerprint(Kind.TOOL, self.name, request),
            invoke=lambda: self.fn(*args, **kwargs),
            encode=to_jsonable,
            decode=decode,
            synth=self._synth,
        )
        return session.intercept(call)

    def _synth(self) -> JsonValue:
        """A stub result for the ``synth`` policy, built from the return annotation."""
        adapter = self._returns
        if adapter is None:
            return None
        try:
            # Validation mode: the stub is validated back into the type, so it must
            # use field names and aliases as the type accepts them.
            return stub_from_schema(adapter.json_schema(mode="validation"))
        except (SynthError, PydanticUserError) as exc:
            raise ReplayError(
                f"tool {self.name!r}: cannot synthesise a result ({exc}); record this call "
                "or replay with --policy fallback"
            ) from exc

    def _decoder(self) -> Callable[[JsonValue], R]:
        adapter = self._returns
        if adapter is None:
            return lambda data: typing.cast(R, data)
        return lambda data: typing.cast(R, adapter.validate_python(data))

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON schema of the parameters, from their annotations."""
        fields: dict[str, Any] = {}
        for param in self._signature.parameters.values():
            default = ... if param.default is param.empty else param.default
            fields[param.name] = (self._hints.get(param.name, Any), default)
        try:
            model = create_model(f"{self.name}_input", **fields)
            schema: dict[str, Any] = model.model_json_schema()
        except PydanticSchemaGenerationError as exc:
            raise ReplayError(
                f"tool {self.name!r}: cannot build an input schema ({exc}); annotate "
                "parameters with JSON-compatible types"
            ) from exc
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema

    def to_anthropic(self) -> ToolParam:
        """The tool definition to pass in ``tools=[...]`` to ``messages.create``."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@overload
def tool[**P, R](fn: Callable[P, R], /) -> Tool[P, R]: ...
@overload
def tool[**P, R](
    *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[P, R]], Tool[P, R]]: ...
def tool[**P, R](
    fn: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool[P, R] | Callable[[Callable[P, R]], Tool[P, R]]:
    """Decorate a tool function so recording and replay sessions intercept it.

    Use bare (``@tool``) or with options (``@tool(name="lookup")``). The function's
    docstring becomes the tool description and its annotated parameters the input
    schema (``Tool.to_anthropic()``).
    """
    if fn is not None:
        return Tool(fn, name=name, description=description)

    def decorate(f: Callable[P, R]) -> Tool[P, R]:
        return Tool(f, name=name, description=description)

    return decorate
