"""What every model-provider adapter implements."""

from __future__ import annotations

from typing import Protocol


class WrappedClient:
    """Marker base class of every adapter's client proxy, so wrapping twice is a no-op."""


class ProviderAdapter(Protocol):
    """Records and replays one model provider's client.

    ``wrap`` returns a proxy that routes the client's model calls through the
    active session (and straight through when none is active). The runner asks
    ``offline_client`` for a stand-in when a replay mode serves model calls and
    no live client was given; the stand-in refuses to be called, so the
    ``fallback`` policy cannot go live by accident.
    """

    name: str

    def accepts(self, client: object) -> bool:
        """Whether ``client`` is this provider's client (sync or async)."""
        ...

    def wrap(self, client: object) -> WrappedClient:
        """A proxy recording ``client``'s model calls."""
        ...

    def owns(self, call_name: str) -> bool:
        """Whether a recorded model envelope called ``call_name`` came from this provider."""
        ...

    def offline_client(self, *, is_async: bool) -> object:
        """An unwrapped client stand-in whose model calls raise; :meth:`wrap` accepts it."""
        ...
