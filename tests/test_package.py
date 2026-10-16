import tomllib
from importlib.metadata import version
from pathlib import Path

import agent_replay

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_version_matches_package_metadata():
    assert agent_replay.__version__ == version("agent-replay")


def test_version_matches_pyproject():
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert agent_replay.__version__ == project["version"]
