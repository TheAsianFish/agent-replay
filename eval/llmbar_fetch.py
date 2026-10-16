"""Re-download the vendored LLMBar files at the pinned commit and verify their sha256s.

LLMBar (Zeng et al., ICLR 2024, https://github.com/princeton-nlp/LLMBar, MIT) is
vendored under ``eval/data/llmbar/`` so the evaluation never needs the network.
This script is how that copy was made and how to check it: it downloads each
file from ``raw.githubusercontent.com`` at :data:`COMMIT` into a fresh temporary
directory, refuses anything whose size or sha256 differs from :data:`FILES`,
checks that each dataset parses as a list of ``{input, output_1, output_2,
label}`` records, and only then moves the files into place. The downloads are
treated as data: nothing from them is executed or imported, and JSON is read
with :mod:`json` only. ``--verify`` checks the vendored copy offline instead.

Standard library only, so it runs isolated::

    python -I eval/llmbar_fetch.py            # re-download and verify (network)
    python -I eval/llmbar_fetch.py --verify   # verify the vendored copy (offline)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REPOSITORY = "princeton-nlp/LLMBar"
COMMIT = "900616bff90b6c6c8e1681f7d079250637c55992"
"""``main`` of the upstream repository on 2026-10-14."""

DEST = Path(__file__).resolve().parent / "data" / "llmbar"
MAX_BYTES = 5_000_000
"""No vendored file is near this; a larger download is not the file we pinned."""

# local path -> (upstream path, sha256, instance count or None for the licence)
FILES: dict[str, tuple[str, str, int | None]] = {
    "Natural/dataset.json": (
        "Dataset/LLMBar/Natural/dataset.json",
        "f7127b7aa78104cb8e03a2b1b3ac0077e95b125714dd441a8ee63666145a3375",
        100,
    ),
    "Adversarial/Neighbor/dataset.json": (
        "Dataset/LLMBar/Adversarial/Neighbor/dataset.json",
        "6712c7eda6342602e0bc7c06c282fa95e2110d4e741a284026a62d937f9780a0",
        134,
    ),
    "Adversarial/GPTInst/dataset.json": (
        "Dataset/LLMBar/Adversarial/GPTInst/dataset.json",
        "1002425f73c2cf54b6e6de99bce3f6c60fa2a17d13194772e37d7481c1351d42",
        92,
    ),
    "Adversarial/GPTOut/dataset.json": (
        "Dataset/LLMBar/Adversarial/GPTOut/dataset.json",
        "1ff56e520d7607750a9866a3be0f5b850392ca4477a1f69d80f32bd3e310f561",
        47,
    ),
    "Adversarial/Manual/dataset.json": (
        "Dataset/LLMBar/Adversarial/Manual/dataset.json",
        "0d9598769a39cdbbd265778c37e49939d41d5f76dcfb17bd75dcfa07516d3837",
        46,
    ),
    "LICENSE-LLMBar": (
        "LICENSE",
        "8872dbf8660b00890b5e07ce1ea1f7a44fa7ae4e1857da56caba172b51dab3cb",
        None,
    ),
}


class FetchError(Exception):
    """A file is missing, the wrong size or hash, or not the expected JSON shape."""


def url(upstream: str) -> str:
    return f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{upstream}"


def parse_records(data: bytes, name: str) -> list[dict[str, str | int]]:
    """The records of one dataset file, after checking every one has the LLMBar shape."""
    try:
        records = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FetchError(f"{name} is not UTF-8 JSON: {exc}") from exc
    if not isinstance(records, list):
        raise FetchError(f"{name} is not a JSON list")
    for i, record in enumerate(records):
        ok = (
            isinstance(record, dict)
            and set(record) == {"input", "output_1", "output_2", "label"}
            and all(isinstance(record[k], str) for k in ("input", "output_1", "output_2"))
            and record["label"] in (1, 2)
            and not isinstance(record["label"], bool)
        )
        if not ok:
            raise FetchError(f"{name}[{i}] is not an {{input, output_1, output_2, label}} record")
    return records


def check(name: str, data: bytes) -> int | None:
    """Verify one file's sha256 (and shape, for a dataset); returns its instance count."""
    _, expected, count = FILES[name]
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise FetchError(f"{name}: sha256 {digest} does not match the pinned {expected}")
    if count is None:
        return None
    found = len(parse_records(data, name))
    if found != count:
        raise FetchError(f"{name}: {found} instances, expected {count}")
    return found


def _download(upstream: str, target: Path) -> bytes:
    with urllib.request.urlopen(url(upstream), timeout=60) as response:  # noqa: S310 - fixed https URL
        data: bytes = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise FetchError(f"{upstream} is larger than {MAX_BYTES} bytes; refusing it")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def fetch(dest: Path = DEST) -> dict[str, int | None]:
    """Download every file into a fresh temporary directory, verify all of them, then copy
    them over ``dest``. Nothing in ``dest`` changes unless every file verifies."""
    counts: dict[str, int | None] = {}
    with tempfile.TemporaryDirectory(prefix="llmbar-download-") as tmp:
        staging = Path(tmp)
        for name, (upstream, _, _) in FILES.items():
            counts[name] = check(name, _download(upstream, staging / name))
        for name in FILES:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging / name, target)
    return counts


def verify(dest: Path = DEST) -> dict[str, int | None]:
    """Check the vendored copy against the pinned hashes, offline."""
    counts: dict[str, int | None] = {}
    for name in FILES:
        path = dest / name
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise FetchError(f"{path} cannot be read ({exc}); run `make data-llmbar`") from exc
        counts[name] = check(name, data)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--dest", type=Path, default=DEST, help=f"default {DEST}")
    parser.add_argument(
        "--verify", action="store_true", help="only verify the files already in --dest (offline)"
    )
    args = parser.parse_args(argv)
    try:
        counts = verify(args.dest) if args.verify else fetch(args.dest)
    except (FetchError, OSError) as exc:
        print(f"llmbar_fetch: {exc}", file=sys.stderr)
        return 1
    for name, count in counts.items():
        print(f"ok  {name}" + (f"  ({count} instances)" if count is not None else ""))
    total = sum(c for c in counts.values() if c is not None)
    print(f"{total} instances at {REPOSITORY}@{COMMIT[:12]} in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
