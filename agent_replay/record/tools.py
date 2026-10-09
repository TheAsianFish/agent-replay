"""The ``@tool`` decorator: record and replay plain Python tool functions."""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Callable
from typing import Any, overload

from anthropic.types import ToolParam
from pydantic import JsonValue, PydanticSchemaGenerationError, TypeAdapter, create_model

from agent_replay._json import fingerprint, to_jsonable
from agent_replay.cassette import Kind
from agent_replay.errors import ReplayError
from agent_replay.session import Call, current_session


class Tool[**P, R]:
    """A tool function that records its calls and can be served from a cassette.

    Calling it behaves exactly like calling the function when no session is
    active. Arguments are recorded by parameter name (defaults applied), and on
    replay the recorded JSON result is validated back into the function's
    annotated return type, so ``-> Order`` returns an ``Order`` both times.
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

    # Type hints are resolved lazily: decorating must never break an import
    # (forward references, annotations pydantic cannot schematise), because
    # outside a session the tool is just the function.
    @functools.cached_property
    def _hints(self) -> dict[str, Any]:
        try:
            return typing.get_type_hints(self.fn)
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
        call: Call[R] = Call(
            kind=Kind.TOOL,
            name=self.name,
            request=request,
            fingerprint=fingerprint(Kind.TOOL, self.name, request),
            invoke=lambda: self.fn(*args, **kwargs),
            encode=to_jsonable,
            decode=decode,
        )
        return session.intercept(call)

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
