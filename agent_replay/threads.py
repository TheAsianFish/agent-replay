"""A thread pool whose workers inherit the submitting code's context.

``concurrent.futures.ThreadPoolExecutor`` runs work in an empty
:mod:`contextvars` context. The session is still found (see
:func:`agent_replay.session.current_session`), but the span of the call that
submitted the work is lost, so tool calls fanned out from inside another tool
are recorded as top-level calls and replay serves them separately from their
parent. Submitting through this executor keeps them nested.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """``ThreadPoolExecutor`` that runs each task in a copy of the submitter's context."""

    def submit[**P, T](self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        context = contextvars.copy_context()
        return super().submit(lambda: context.run(fn, *args, **kwargs))
