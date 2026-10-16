"""The release workflow publishes only a final, matching tag on main, from this repository."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text())
GUARD = (
    "github.repository == 'TheAsianFish/agent-replay' && github.event_name == 'push' "
    "&& startsWith(github.ref, 'refs/tags/v')"
)
PUBLISHING = {"publish-pypi", "github-release", "docker"}


def test_every_publishing_job_is_guarded_and_needs_the_build():
    for name in PUBLISHING:
        job = WORKFLOW["jobs"][name]
        assert job["if"] == GUARD, name
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert "build" in needs, name
    assert set(WORKFLOW["jobs"]) == {"build", *PUBLISHING}


def test_write_permissions_are_scoped_to_the_job_that_needs_them():
    assert WORKFLOW["permissions"] == {"contents": "read"}
    assert WORKFLOW["jobs"]["publish-pypi"]["permissions"] == {"id-token": "write"}
    assert WORKFLOW["jobs"]["github-release"]["permissions"] == {"contents": "write"}
    assert WORKFLOW["jobs"]["docker"]["permissions"] == {"contents": "read", "packages": "write"}
    assert "permissions" not in WORKFLOW["jobs"]["build"]


def test_no_run_block_interpolates_expressions():
    for name, job in WORKFLOW["jobs"].items():
        for step in job["steps"]:
            assert "${{" not in step.get("run", ""), (name, step.get("name"))


def test_the_version_step_refuses_prereleases_mismatches_and_tags_off_main():
    build = WORKFLOW["jobs"]["build"]
    assert build["steps"][0]["with"] == {"fetch-depth": 0}
    step = next(s for s in build["steps"] if s.get("id") == "version")
    run = step["run"]
    assert '[[ "$version" =~ ^[0-9]+\\.[0-9]+\\.[0-9]+$ ]]' in run
    assert '[ "${GITHUB_REF_NAME#v}" != "$version" ]' in run
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in run
    # Each refusal exits 1 only on a tag; a manual run reports the version and goes on.
    assert run.count("exit 1") == 3
    assert '[ "$IS_TAG" = "true" ] || exit 0' in run


def test_the_version_regex_accepts_only_final_versions():
    pattern = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
    assert pattern.match("1.0.0")
    assert pattern.match("10.20.30")
    for version in ("1.0.0-rc1", "1.0.0rc1", "1.0", "v1.0.0", "1.0.0.post1", "1.0.0 "):
        assert not pattern.match(version), version
