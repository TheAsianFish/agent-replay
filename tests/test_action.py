"""The composite GitHub Action: action.yml's wiring and its helper script, offline."""

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parents[1]
ACTION = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
ACTION_TEXT = (ROOT / "action.yml").read_text(encoding="utf-8")
SUITE = "examples/retail_agent/replay.toml"
MARKER = f"<!-- agent-replay: {SUITE} -->"


def _load_helper() -> Any:
    spec = importlib.util.spec_from_file_location(
        "replay_action", ROOT / "action" / "replay_action.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_action"] = module
    spec.loader.exec_module(module)
    return module


ra = _load_helper()


# action.yml


def test_action_is_composite_with_marketplace_metadata():
    assert ACTION["runs"]["using"] == "composite"
    assert ACTION["name"]
    assert len(ACTION["description"]) < 125
    assert ACTION["author"]
    assert ACTION["branding"]["icon"]
    assert ACTION["branding"]["color"]
    assert ACTION["inputs"]["suite"]["required"] is True


def test_every_input_is_described_and_every_reference_declared():
    for name, spec in ACTION["inputs"].items():
        assert spec.get("description"), name
        assert "default" in spec or spec.get("required"), name
    referenced = set(re.findall(r"\$\{\{\s*inputs\.([A-Za-z0-9_-]+)", ACTION_TEXT))
    assert referenced <= set(ACTION["inputs"])
    assert set(ACTION["inputs"]) <= referenced  # and none is declared but ignored


def test_no_run_block_interpolates_expressions():
    for step in ACTION["runs"]["steps"]:
        if "run" in step:
            assert "${{" not in step["run"], step["name"]
            assert step["shell"] == "bash"


def test_every_output_maps_to_a_step_output(tmp_path):
    steps = {s["id"]: s for s in ACTION["runs"]["steps"] if "id" in s}
    summarize = ra.summarize(tmp_path / "r.json", tmp_path / "r.md", 0)
    comment = ra.comment(
        suite=SUITE,
        markdown_path=tmp_path / "r.md",
        exit_code=0,
        body_path=tmp_path / "c.md",
        post=False,
        dry_run=False,
        env={},
    )
    written = {"summarize": set(summarize), "comment": set(comment)}
    for name, spec in ACTION["outputs"].items():
        assert spec["description"], name
        match = re.fullmatch(r"\$\{\{ steps\.([a-z]+)\.outputs\.([a-z-]+) \}\}", spec["value"])
        assert match, name
        step_id, key = match.groups()
        assert step_id in steps, name
        if step_id in written:
            assert key in written[step_id], name
        else:
            assert f"{key}=" in steps[step_id]["run"], name


def test_the_example_workflow_uses_only_declared_inputs():
    workflow = yaml.safe_load(
        (ROOT / "examples" / "github-workflow" / "replay.yml").read_text(encoding="utf-8")
    )
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "write"}
    uses = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("TheAsianFish/agent-replay@")
    ]
    assert uses
    for step in uses:
        assert set(step["with"]) <= set(ACTION["inputs"])
        assert "suite" in step["with"]


def test_the_ci_job_uses_only_declared_inputs():
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    steps = [s for s in ci["jobs"]["action"]["steps"] if s.get("uses") == "./"]
    assert len(steps) == 2
    for step in steps:
        assert set(step["with"]) <= set(ACTION["inputs"])
        assert step["with"]["dry-run"] is True


# summarize


def _report(tmp_path, *passed):
    scenarios = [
        {"name": f"s{i}", "passed": p, "ran": True, "error": None, "checks": []}
        for i, p in enumerate(passed)
    ]
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"suite": SUITE, "passed": all(passed), "scenarios": scenarios}))
    (tmp_path / "report.md").write_text("## summary\n")
    return path


def _outputs(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def _summarize(tmp_path, exit_code):
    out = tmp_path / "out"
    code = ra.main(
        [
            "summarize",
            "--report-json",
            str(tmp_path / "report.json"),
            "--report-markdown",
            str(tmp_path / "report.md"),
            "--exit-code",
            str(exit_code),
        ],
        env={"GITHUB_OUTPUT": str(out)},
    )
    assert code == 0
    return _outputs(out)


def test_summarize_pass(tmp_path):
    _report(tmp_path, True, True, True)
    assert _summarize(tmp_path, 0) == {
        "passed": "true",
        "scenarios": "3",
        "failed": "0",
        "report-json": str(tmp_path / "report.json"),
        "report-markdown": str(tmp_path / "report.md"),
    }


def test_summarize_fail(tmp_path):
    _report(tmp_path, True, False, False)
    outputs = _summarize(tmp_path, 1)
    assert (outputs["passed"], outputs["scenarios"], outputs["failed"]) == ("false", "3", "2")


def test_summarize_broken_without_a_report(tmp_path, capsys):
    outputs = _summarize(tmp_path, 2)
    assert outputs == {
        "passed": "false",
        "scenarios": "0",
        "failed": "0",
        "report-json": "",
        "report-markdown": "",
    }
    assert "::error::replay ci exited 2 without a report" in capsys.readouterr().out


def test_summarize_unparseable_report_warns(tmp_path, capsys):
    (tmp_path / "report.json").write_text("{not json")
    outputs = _summarize(tmp_path, 2)
    assert outputs["scenarios"] == "0"
    assert outputs["report-json"] == ""
    assert "::warning::cannot read the replay report" in capsys.readouterr().out


def test_multiline_outputs_use_a_delimiter(tmp_path):
    out = tmp_path / "out"
    ra.write_outputs({"a": "one\ntwo", "b": "x"}, {"GITHUB_OUTPUT": str(out)})
    lines = out.read_text().splitlines()
    assert lines[0].startswith("a<<EOF_")
    assert lines[1:4] == ["one", "two", lines[0].split("<<")[1]]
    assert lines[4] == "b=x"


# comment body


RUN_ENV = {
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "owner/repo",
    "GITHUB_RUN_ID": "42",
}


def test_body_has_marker_summary_and_run_link():
    body = ra.build_body(SUITE, "## ✅ Replay: 6/6\n", 0, ra.run_url(RUN_ENV))
    assert body.startswith(MARKER + "\n")
    assert "## ✅ Replay: 6/6" in body
    assert "(https://github.com/owner/repo/actions/runs/42)" in body


def test_body_without_a_summary_says_the_suite_broke():
    body = ra.build_body(SUITE, None, 2, None)
    assert body.startswith(MARKER)
    assert "could not run" in body
    assert "exited 2" in body


def test_marker_cannot_close_the_html_comment():
    assert ra.marker("a-->b\nc").count("-->") == 1


def test_long_bodies_are_truncated_below_the_limit_with_fences_closed():
    markdown = "## head\n\n  ```diff\n" + "".join(f"  - line {i}\n" for i in range(20_000))
    body = ra.build_body(SUITE, markdown, 1, ra.run_url(RUN_ENV))
    assert len(body) <= ra.MAX_BODY < ra.COMMENT_LIMIT
    assert body.startswith(MARKER)
    assert body.rstrip().endswith("/actions/runs/42)")
    assert "Summary truncated" in body
    fences = [line for line in body.splitlines() if line.strip().startswith("```")]
    assert len(fences) % 2 == 0


def test_short_bodies_are_untouched():
    markdown = "x" * 1000 + "\n"
    assert markdown in ra.build_body(SUITE, markdown, 0, None)


# upsert


class FakeGitHub:
    def __init__(self, pages: list[list[dict[str, Any]]], status: int = 200) -> None:
        self.pages = pages
        self.status = status
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def __call__(self, method, url, payload):
        self.calls.append((method, url, payload))
        if self.status != 200:
            return ra.Response(self.status, {"message": "Resource not accessible by integration"})
        if method == "GET":
            page = int(re.search(r"[?&]page=(\d+)", url).group(1))
            return ra.Response(200, self.pages[page - 1] if page <= len(self.pages) else [])
        if method == "PATCH":
            return ra.Response(200, {"html_url": "https://github.com/c/updated"})
        return ra.Response(201, {"html_url": "https://github.com/c/created"})


@pytest.fixture
def pr_env(tmp_path):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7}}))
    return {
        **RUN_ENV,
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_TOKEN": "ghs_secret_token",
        "GITHUB_API_URL": "https://api.github.com",
    }


def _comment(tmp_path, env, http, *, post=True, dry_run=False):
    (tmp_path / "report.md").write_text("## summary\n")
    return ra.comment(
        suite=SUITE,
        markdown_path=tmp_path / "report.md",
        exit_code=0,
        body_path=tmp_path / "comment.md",
        post=post,
        dry_run=dry_run,
        env=env,
        http=http,
    )


def _other(i):
    return {"id": i, "body": f"comment {i}"}


def test_creates_the_comment_when_absent(tmp_path, pr_env):
    gh = FakeGitHub([[_other(1), _other(2)]])
    outputs = _comment(tmp_path, pr_env, gh)
    assert outputs["comment-url"] == "https://github.com/c/created"
    method, url, payload = gh.calls[-1]
    assert (method, url) == ("POST", "https://api.github.com/repos/owner/repo/issues/7/comments")
    assert payload["body"].startswith(MARKER)
    assert (tmp_path / "comment.md").read_text() == payload["body"]


def test_updates_the_marker_comment_found_on_page_two(tmp_path, pr_env):
    quoted = {"id": 5, "body": f"> {MARKER}\nquoting it does not count"}
    other_suite = {"id": 6, "body": "<!-- agent-replay: other/replay.toml -->\n"}
    page1 = [quoted, other_suite, *(_other(i) for i in range(10, 108))]
    page2 = [_other(200), {"id": 999, "body": MARKER + "\nold"}]
    gh = FakeGitHub([page1, page2])
    outputs = _comment(tmp_path, pr_env, gh)
    assert outputs["comment-url"] == "https://github.com/c/updated"
    assert [m for m, _, _ in gh.calls] == ["GET", "GET", "PATCH"]
    assert gh.calls[-1][1] == "https://api.github.com/repos/owner/repo/issues/comments/999"


def test_forbidden_warns_without_failing_and_hides_the_token(tmp_path, pr_env, capsys):
    outputs = _comment(tmp_path, pr_env, FakeGitHub([], status=403))
    assert outputs["comment-url"] == ""
    out = capsys.readouterr().out
    assert "::warning::agent-replay did not comment on #7" in out
    assert "HTTP 403" in out
    assert "pull-requests: write" in out
    assert "ghs_secret_token" not in out


def test_non_pr_events_do_not_comment(tmp_path, pr_env):
    gh = FakeGitHub([])
    outputs = _comment(tmp_path, {**pr_env, "GITHUB_EVENT_NAME": "push"}, gh)
    assert gh.calls == []
    assert outputs["comment-url"] == ""
    assert (tmp_path / "comment.md").read_text().startswith(MARKER)


def test_comment_false_writes_the_body_but_does_not_post(tmp_path, pr_env):
    gh = FakeGitHub([])
    outputs = _comment(tmp_path, pr_env, gh, post=False)
    assert gh.calls == []
    assert Path(outputs["comment-body"]).read_text().startswith(MARKER)


def test_dry_run_prints_the_body_without_calling_the_api(tmp_path, pr_env, capsys):
    gh = FakeGitHub([])
    outputs = _comment(tmp_path, pr_env, gh, dry_run=True)
    assert gh.calls == []
    assert outputs["comment-url"] == ""
    out = capsys.readouterr().out
    assert "PR comment for #7: dry run, not posted" in out
    assert MARKER in out
    stop = re.search(r"::stop-commands::(\w+)", out).group(1)
    assert f"::{stop}::" in out


def test_pr_target_events_and_missing_token(tmp_path, pr_env, capsys):
    env = {**pr_env, "GITHUB_EVENT_NAME": "pull_request_target", "GITHUB_TOKEN": ""}
    gh = FakeGitHub([])
    _comment(tmp_path, env, gh)
    assert gh.calls == []
    assert "::warning::agent-replay did not comment: no github-token" in capsys.readouterr().out


def test_the_real_http_layer_refuses_plain_http():
    with pytest.raises(ra.CommentError, match="non-HTTPS"):
        ra.urllib_http("t")("GET", "http://example.com", None)


def test_dry_run_prints_the_body_even_without_a_pr_number(tmp_path, pr_env, capsys):
    Path(pr_env["GITHUB_EVENT_PATH"]).write_text("{}")
    _comment(tmp_path, pr_env, FakeGitHub([]), dry_run=True)
    out = capsys.readouterr().out
    assert "::warning::agent-replay did not comment: the event payload has no pull" in out
    assert "(no pull request to post to): dry run, not posted" in out
