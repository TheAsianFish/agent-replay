"""Shared fixtures. Every test except those marked ``live`` runs with sockets blocked.

Tests marked ``postgres`` may reach the host of ``REPLAY_TEST_PG_URL`` and nothing else.
The guard patches Python's socket layer; psycopg resolves names through it, so a
hostname is enforced, while a numeric host would let libpq connect natively.
"""

from __future__ import annotations

import os
import socket
import warnings
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy.engine import make_url


class NetworkBlockedError(RuntimeError):
    pass


def _blocked(*_args: object, **_kwargs: object) -> None:
    raise NetworkBlockedError(
        "tests must not touch the network; mark live tests with @pytest.mark.live"
    )


def _host_of(address: object) -> object:
    return address[0] if isinstance(address, tuple) else address


def _only(hosts: set[str], real: Callable[..., Any], position: int) -> Callable[..., Any]:
    def guarded(*args: Any, **kwargs: Any) -> Any:
        address = args[position] if len(args) > position else kwargs.get("host")
        if _host_of(address) in hosts:
            return real(*args, **kwargs)
        return _blocked()

    return guarded


def _pg_hosts() -> set[str]:
    raw = os.environ.get("REPLAY_TEST_PG_URL")
    host = make_url(raw).host if raw else None
    if not host:
        return set()
    return {host, "127.0.0.1", "::1"} if host == "localhost" else {host}


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live") is not None:
        return
    hosts = _pg_hosts() if request.node.get_closest_marker("postgres") is not None else set()
    if hosts:
        monkeypatch.setattr(socket.socket, "connect", _only(hosts, socket.socket.connect, 1))
        monkeypatch.setattr(socket.socket, "connect_ex", _only(hosts, socket.socket.connect_ex, 1))
        monkeypatch.setattr(socket, "create_connection", _only(hosts, socket.create_connection, 0))
        monkeypatch.setattr(socket, "getaddrinfo", _only(hosts, socket.getaddrinfo, 0))
        return
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


@pytest.fixture(autouse=True)
def _recorded_live_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pydantic-ai example's committed cassettes were recorded on its default model;
    an exported ``REPLAY_OLLAMA_MODEL`` would make every replay of them approximate."""
    monkeypatch.delenv("REPLAY_OLLAMA_MODEL", raising=False)


def _sk_kappa(a: Sequence[int], b: Sequence[int], *, weights: Any = None) -> float:
    from sklearn.metrics import cohen_kappa_score  # noqa: PLC0415 - dev dependency, oracle only

    with warnings.catch_warnings():  # undefined kappa warns, and returns NaN as wanted
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(list(a), list(b), labels=[0, 1], weights=weights))


@pytest.fixture
def core_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    """``REPLAY_STATS_IMPL=core`` with a real kappa (scikit-learn's) until ``stats.py`` is
    written; the interval and gate come from the baseline. Lets a test reach a report that
    may gate, which a ``baseline`` report never does."""
    from agent_replay import stats_baseline, stats_impl  # noqa: PLC0415

    shim = SimpleNamespace(
        cohens_kappa=_sk_kappa,
        bootstrap_ci=stats_baseline.bootstrap_ci,
        wilson_interval=stats_baseline.wilson_interval,
        SequentialGate=stats_baseline.SequentialGate,
    )
    monkeypatch.setitem(stats_impl._MODULES, "core", cast(Any, shim))
    monkeypatch.setenv("REPLAY_STATS_IMPL", "core")
