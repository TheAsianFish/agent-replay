"""scripts/release_notes.py: the release workflow's CHANGELOG.md section extraction."""

import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "release_notes.py"
_spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
release_notes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [Unreleased]

## [1.1.0] - 2026-11-02

### Fixed

- Something.

## [1.0.0] - 2026-10-20

The first release.

### Added

- Everything.

[Unreleased]: https://github.com/o/r/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/o/r/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/o/r/releases/tag/v1.0.0
"""


def test_extracts_a_middle_section():
    assert release_notes.release_notes(CHANGELOG, "1.1.0") == "### Fixed\n\n- Something.\n"


def test_extracts_the_last_section_without_link_references():
    notes = release_notes.release_notes(CHANGELOG, "1.0.0")
    assert notes == "The first release.\n\n### Added\n\n- Everything.\n"


@pytest.mark.parametrize(
    ("changelog", "message"),
    [
        (CHANGELOG, "has no '## [2.0.0] - YYYY-MM-DD' section"),
        (CHANGELOG.replace("[2.0.0", "[x"), "has no '## [2.0.0] - YYYY-MM-DD' section"),
        ("## [2.0.0] - Unreleased\n\n- x\n", "is not dated (Unreleased)"),
        ("## [2.0.0]\n\n- x\n", "is not dated (no date)"),
        ("## [2.0.0] - soon\n\n- x\n", "'soon' is not YYYY-MM-DD"),
        ("## [2.0.0] - 2026-10-20\n\n## [1.0.0] - 2026-10-01\n", "section is empty"),
    ],
)
def test_refuses_a_section_that_cannot_ship(changelog: str, message: str):
    with pytest.raises(release_notes.NotesError, match=re.escape(message)):
        release_notes.release_notes(changelog, "2.0.0")


def test_cli_strips_the_tag_prefix_and_exits_1_with_the_fix(tmp_path: Path, capsys):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(CHANGELOG, encoding="utf-8")
    out = tmp_path / "notes.md"
    assert release_notes.main(["v1.1.0", "--changelog", str(path), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "### Fixed\n\n- Something.\n"
    assert release_notes.main(["3.0.0", "--changelog", str(path)]) == 1
    assert "add the release's notes before tagging" in capsys.readouterr().err
