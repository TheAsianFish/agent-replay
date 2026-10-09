"""Shared fixtures. Every test except those marked ``live`` runs with sockets blocked."""

from __future__ import annotations

import socket

import pytest


class NetworkBlockedError(RuntimeError):
    pass


def _blocked(*_args: object, **_kwargs: object) -> None:
    raise NetworkBlockedError(
        "tests must not touch the network; mark live tests with @pytest.mark.live"
    )


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setattr(socket.socket, "connect", _blocked)
        monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)
        monkeypatch.setattr(socket, "getaddrinfo", _blocked)
