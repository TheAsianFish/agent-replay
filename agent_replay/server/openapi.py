"""Write the API's OpenAPI schema; the dashboard generates its TypeScript types from it.

``python -m agent_replay.server.openapi dashboard/openapi.json``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_replay.server.app import create_app


def schema_json() -> str:
    return json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m agent_replay.server.openapi OUT.json")
    Path(args[0]).write_text(schema_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
