from importlib.metadata import version

import agent_replay


def test_version_matches_package_metadata():
    assert agent_replay.__version__ == version("agent-replay")
