"""Print the CHANGELOG.md section for one release, the notes of its GitHub Release.

    python3 scripts/release_notes.py VERSION [--changelog CHANGELOG.md] [--out notes.md]

The section is the one headed ``## [VERSION] - YYYY-MM-DD`` (Keep a Changelog). It
exits 1 with the fix when that heading is missing, still reads ``Unreleased`` or has
no date, or the section is empty, so a tag cannot ship without dated notes.
Stdlib only.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\](?:\s*-\s*(?P<date>.*?))?\s*$")


class NotesError(Exception):
    """The changelog has no releasable section for the version."""


def release_notes(changelog: str, version: str) -> str:
    """The body of ``version``'s section, without its heading."""
    lines = changelog.splitlines()
    start: int | None = None
    released = ""
    for i, line in enumerate(lines):
        match = HEADING.match(line)
        if match and match["version"] == version:
            start, released = i + 1, (match["date"] or "").strip()
            break
    if start is None:
        raise NotesError(
            f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section; "
            "add the release's notes before tagging it"
        )
    if not released or released.lower() == "unreleased":
        raise NotesError(
            f"CHANGELOG.md's [{version}] section is not dated ({released or 'no date'}); "
            f"set its heading to '## [{version}] - YYYY-MM-DD' before tagging"
        )
    try:
        date.fromisoformat(released)
    except ValueError:
        raise NotesError(
            f"CHANGELOG.md's [{version}] date '{released}' is not YYYY-MM-DD; fix the heading"
        ) from None
    end = next(
        (i for i in range(start, len(lines)) if HEADING.match(lines[i])),
        len(lines),
    )
    # Drop the link references Keep a Changelog keeps at the bottom of the file.
    body = [line for line in lines[start:end] if not re.match(r"^\[[^\]]+\]:\s*\S", line)]
    text = "\n".join(body).strip()
    if not text:
        raise NotesError(f"CHANGELOG.md's [{version}] section is empty; write its notes")
    return text + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("version", help="Release version, without the leading v (e.g. 1.0.0).")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--out", type=Path, help="Write the notes here instead of stdout.")
    args = parser.parse_args(argv)
    try:
        text = args.changelog.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"release_notes: cannot read {args.changelog}: {exc}", file=sys.stderr)
        return 1
    try:
        notes = release_notes(text, args.version.removeprefix("v"))
    except NotesError as exc:
        print(f"release_notes: {exc}", file=sys.stderr)
        return 1
    if args.out is None:
        sys.stdout.write(notes)
    else:
        args.out.write_text(notes, encoding="utf-8")
        print(f"release_notes: {args.version} notes written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
