"""Shared fixtures. Every test except those marked ``live`` runs with sockets blocked.

Tests marked ``postgres`` may reach the host of ``REPLAY_TEST_PG_URL`` and nothing else.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from typing import Any

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
        if _host_of(args[position]) in hosts:
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
