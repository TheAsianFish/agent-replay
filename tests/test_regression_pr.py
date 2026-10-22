"""``make regression-pr`` (eval/regression_pr.py): the real-PR demo, with a fake ``gh``."""

import importlib.util
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from eval import regression_pr as rp
from examples.retail_agent import mutants

ROOT = Path(__file__).parents[1]
REPO = "owner/agent-replay"
PR_URL = f"https://github.com/{REPO}/pull/42"
NOW = datetime(2026, 10, 21, 12, 0, 0, tzinfo=UTC)
HEAD = "regression-demo/20261021-120000"
BASE = "regression-demo/base-20261021-120000"

REPORT = {
    "passed": False,
    "duration_ms": 240,
    "scenarios": [
        {
            "name": "cancel_pending_laptop",
            "passed": False,
            "checks": [
                {"check": "tool_sequence", "passed": True},
                {"check": "rule:matches_ground_truth", "passed": False},
            ],
        },
        {"name": "lookup_latest_order_total", "passed": True, "checks": []},
    ],
}
COMMENT_BODY = rp.marker() + "\n## ❌ Replay: 1/2 scenarios passed\n"


class FakeShell(rp.Shell):
    """Answers the git and gh commands Demo runs; records every call in order."""

    def __init__(self, *, conclusion: str = "failure", logged_in: bool = True) -> None:
        super().__init__()
        self.conclusion = conclusion
        self.logged_in = logged_in
        self.run_polls = 0

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> str:
        self.log.append(list(argv))
        args = list(argv)
        if args[:3] == ["gh", "auth", "status"]:
            if not self.logged_in:
                raise rp.RegressionPrError("`gh auth status` exited 1: not logged in")
            return ""
        if args == ["git", "rev-parse", "HEAD"]:
            return "feedface00112233\n"
        if args[:2] == ["git", "rev-parse"]:
            return "abc123def4567890\n"
        if args[:3] == ["git", "remote", "get-url"]:
            return f"git@github.com:{REPO}.git\n"
        if args[:2] == ["git", "show"]:
            return (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        if args[:3] == ["git", "worktree", "add"]:
            worktree = Path(args[5])
            (worktree / Path(rp.SUITE).parent).mkdir(parents=True)
            (worktree / rp.SUITE).write_text((ROOT / rp.SUITE).read_text(encoding="utf-8"))
            return ""
        if args[:3] == ["gh", "pr", "create"]:
            return "Creating pull request\n" + PR_URL + "\n"
        if args[:3] == ["gh", "run", "list"]:
            self.run_polls += 1
            status = "completed" if self.run_polls >= 2 else "in_progress"
            assert args[args.index("--commit") + 1] == "feedface00112233"
            return json.dumps(
                [
                    {"databaseId": 98, "status": "completed", "workflowName": "CI"},
                    {"databaseId": 99, "status": status, "workflowName": rp.WORKFLOW_NAME},
                ]
            )
        if args[:3] == ["gh", "run", "view"]:
            return json.dumps(
                {
                    "url": f"https://github.com/{REPO}/actions/runs/99",
                    "headSha": "feedface00112233",
                    "conclusion": self.conclusion,
                    "createdAt": "2026-10-21T12:00:05Z",
                    "updatedAt": "2026-10-21T12:01:15Z",
                    "jobs": [
                        {
                            "name": rp.JOB_NAME,
                            "url": f"https://github.com/{REPO}/actions/runs/99/job/7",
                            "conclusion": self.conclusion,
                            "startedAt": "2026-10-21T12:00:10Z",
                            "completedAt": "2026-10-21T12:01:02Z",
                        }
                    ],
                }
            )
        if args[:3] == ["gh", "run", "download"]:
            target = Path(args[args.index("-D") + 1]) / "replay-report-replay-x"
            target.mkdir(parents=True)
            (target / "report.json").write_text(json.dumps(REPORT))
            return ""
        if args[:2] == ["gh", "api"] and args[-1].endswith("/issues/42/comments"):
            return json.dumps(
                [
                    # Anyone can post the marker; only the bot's comment counts.
                    {"id": 1, "user": {"login": "someone"}, "body": COMMENT_BODY},
                    {
                        "id": 2,
                        "user": {"login": "github-actions[bot]"},
                        "body": COMMENT_BODY,
                        "html_url": f"{PR_URL}#issuecomment-2",
                        "created_at": "2026-10-21T12:01:00Z",
                    },
                ]
            )
        if args[:2] == ["gh", "api"] and args[-1].endswith("/issues/comments/2"):
            return json.dumps({"body_html": "<h2>❌ Replay: 1/2 scenarios passed</h2>"})
        if args[:1] == ["npm"]:
            Path(args[-1]).write_bytes(b"\x89PNG fake")
            return ""
        return ""


def _demo(tmp_path: Path, shell: FakeShell) -> rp.Demo:
    return rp.Demo(
        shell,
        sleep=lambda _s: None,
        now=lambda: NOW,
        out=tmp_path / "out",
        screenshot=tmp_path / "pr-comment.png",
        poll_s=1.0,
        timeout_s=5.0,
    )


def _index(log: list[list[str]], prefix: list[str]) -> int:
    for i, argv in enumerate(log):
        if argv[: len(prefix)] == prefix:
            return i
    raise AssertionError(f"{prefix} was never run")


def test_seed_regression_swaps_only_the_default_agent() -> None:
    text = (ROOT / rp.SUITE).read_text(encoding="utf-8")
    seeded = rp.seed_regression(text, "wrong_write_argument")
    assert 'agent = "examples.retail_agent.mutants:wrong_write_argument"' in seeded
    assert rp.DEFAULT_AGENT not in seeded
    assert len(seeded.splitlines()) == len(text.splitlines())


def test_seed_regression_refuses_a_suite_it_does_not_recognise() -> None:
    with pytest.raises(rp.RegressionPrError, match="exactly once"):
        rp.seed_regression("[defaults]\nagent = 'x:y'\n", "first_tool_only")


def test_seeded_mutant_exists() -> None:
    assert callable(getattr(mutants, rp.DEFAULT_MUTANT))


def test_marker_matches_the_action() -> None:
    spec = importlib.util.spec_from_file_location(
        "replay_action_for_regression_pr", ROOT / "action" / "replay_action.py"
    )
    assert spec is not None
    assert spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert rp.marker() == helper.marker(rp.SUITE)


def test_scenario_outcomes_names_failed_checks() -> None:
    assert rp.scenario_outcomes(REPORT) == {
        "failed": [
            {"name": "cancel_pending_laptop", "failed_checks": ["rule:matches_ground_truth"]}
        ],
        "passed": ["lookup_latest_order_total"],
    }


def test_pick_job_names_the_jobs_it_saw() -> None:
    with pytest.raises(rp.RegressionPrError, match="'other'"):
        rp.pick_job({"jobs": [{"name": "other"}]})


def test_pick_comment_needs_the_bot() -> None:
    impostor = {"id": 1, "user": {"login": "someone"}, "body": COMMENT_BODY}
    with pytest.raises(rp.RegressionPrError, match="pull-requests: write"):
        rp.pick_comment([impostor])


def test_full_run_opens_waits_captures_and_closes(tmp_path: Path) -> None:
    shell = FakeShell()
    evidence = _demo(tmp_path, shell).run()

    assert evidence["pr_url"] == PR_URL
    assert evidence["conclusion"] == "failure"
    assert evidence["check_url"].endswith("/job/7")
    assert evidence["comment_url"] == f"{PR_URL}#issuecomment-2"
    assert evidence["job_wall_seconds"] == 52.0
    assert evidence["run_wall_seconds"] == 70.0
    assert evidence["scenarios"]["failed"][0]["name"] == "cancel_pending_laptop"
    assert shell.run_polls == 2

    out = tmp_path / "out"
    assert json.loads((out / "evidence.json").read_text()) == evidence
    assert (out / "comment.md").read_text() == COMMENT_BODY
    assert json.loads((out / "report.json").read_text()) == REPORT

    log = shell.log
    create = log[_index(log, ["gh", "pr", "create"])]
    assert create[create.index("--base") + 1] == BASE
    assert create[create.index("--head") + 1] == HEAD
    assert ["git", "push", "--quiet", "origin", f"abc123def4567890:refs/heads/{BASE}"] in log
    shot = log[_index(log, ["npm"])]
    assert shot[shot.index("screenshot") + 3].startswith("file://")
    # Evidence first, then the PR is closed and both branches deleted.
    close = _index(log, ["gh", "pr", "close"])
    assert _index(log, ["gh", "run", "download"]) < close
    assert "--delete-branch" in log[close]
    assert ["git", "push", "--quiet", "origin", "--delete", BASE] in log
    assert ["git", "branch", "-D", HEAD] in log
    assert (tmp_path / "pr-comment.png").exists()


def test_a_green_check_is_a_failed_demo_and_keeps_old_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "evidence.json").write_text("previous")
    shell = FakeShell(conclusion="success")
    with pytest.raises(rp.RegressionPrError, match="did not stop"):
        _demo(tmp_path, shell).run()
    assert (out / "evidence.json").read_text() == "previous"
    assert not (tmp_path / "pr-comment.png").exists()
    # Left open to inspect, with the commands that clean it up; the worktree is gone.
    err = capsys.readouterr().err
    assert f"gh pr close {PR_URL}" in err
    assert f"git push origin --delete {BASE} {HEAD}" in err
    assert not any(argv[:3] == ["gh", "pr", "close"] for argv in shell.log)
    assert ["git", "branch", "-D", HEAD] in shell.log


def test_a_failure_before_the_pr_still_names_the_branches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class NoPr(FakeShell):
        def run(self, argv: Sequence[str], **kwargs: Any) -> str:
            if list(argv[:3]) == ["gh", "pr", "create"]:
                raise rp.RegressionPrError("`gh pr create` exited 1: no permission")
            return super().run(argv, **kwargs)

    with pytest.raises(rp.RegressionPrError, match="no permission"):
        _demo(tmp_path, NoPr()).run()
    assert f"git push origin --delete {BASE} {HEAD}" in capsys.readouterr().err


def test_an_unknown_mutant_stops_before_any_push(tmp_path: Path) -> None:
    shell = FakeShell()
    demo = _demo(tmp_path, shell)
    demo.mutant = "no_such_mutant"
    with pytest.raises(rp.RegressionPrError, match="no mutant 'no_such_mutant'"):
        demo.run()
    assert not any(argv[:2] == ["git", "push"] for argv in shell.log)


def test_a_broken_suite_is_not_a_caught_regression() -> None:
    broken = {
        "passed": False,
        "scenarios": [{"name": "a", "passed": False, "ran": False, "error": "ImportError: x"}],
    }
    with pytest.raises(rp.RegressionPrError, match="broken suite"):
        rp.check_regression(broken)
    with pytest.raises(rp.RegressionPrError, match="no failed scenario"):
        rp.check_regression({"passed": True, "scenarios": [{"name": "a", "passed": True}]})
    rp.check_regression(REPORT)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/agent-replay.git",
        "https://github.com/owner/agent-replay",
        "git@github.com:owner/agent-replay.git",
    ],
)
def test_origin_repo(url: str) -> None:
    assert rp.origin_repo(url) == REPO


def test_origin_repo_refuses_other_hosts() -> None:
    with pytest.raises(rp.RegressionPrError, match="not a GitHub repository"):
        rp.origin_repo("https://gitlab.com/owner/agent-replay.git")


def test_not_logged_in_names_the_fix(tmp_path: Path) -> None:
    shell = FakeShell(logged_in=False)
    with pytest.raises(rp.RegressionPrError, match="gh auth login"):
        _demo(tmp_path, shell).run()
    assert not any(argv[:2] == ["git", "push"] for argv in shell.log)


def test_a_missing_browser_stops_before_any_push(tmp_path: Path) -> None:
    class NoBrowser(FakeShell):
        def run(self, argv: Sequence[str], **kwargs: Any) -> str:
            if list(argv[:1]) == ["npm"]:
                raise rp.RegressionPrError("`npm exec playwright` exited 1: no chromium")
            return super().run(argv, **kwargs)

    shell = NoBrowser()
    with pytest.raises(rp.RegressionPrError, match="playwright install chromium"):
        _demo(tmp_path, shell).run()
    assert not any(argv[:2] == ["git", "push"] for argv in shell.log)
    assert not any(argv[:3] == ["gh", "pr", "create"] for argv in shell.log)


def test_wait_for_run_times_out_with_a_command_to_watch(tmp_path: Path) -> None:
    class Stuck(FakeShell):
        def run(self, argv: Sequence[str], **kwargs: Any) -> str:
            if list(argv[:3]) == ["gh", "run", "list"]:
                return json.dumps(
                    [{"databaseId": 1, "status": "queued", "workflowName": rp.WORKFLOW_NAME}]
                )
            return super().run(argv, **kwargs)

    with pytest.raises(rp.RegressionPrError, match="queued after 5 s"):
        _demo(tmp_path, Stuck()).wait_for_run(REPO, HEAD, "feedface00112233")


def test_results_markdown_quotes_the_evidence(tmp_path: Path) -> None:
    evidence = _demo(tmp_path, FakeShell()).run()
    text = rp.results_markdown(evidence)
    assert PR_URL in text
    assert "/job/7" in text
    assert "#issuecomment-2" in text
    assert "52 s for the job" in text
    assert "1 of 2" in text
    assert "`cancel_pending_laptop`: rule:matches_ground_truth" in text


def test_comment_page_embeds_githubs_html() -> None:
    page = rp.comment_page(
        {"html_url": f"{PR_URL}#issuecomment-2", "body_html": "<h2>x</h2>", "created_at": "t"},
        PR_URL,
    )
    assert "<h2>x</h2>" in page
    assert "#42" in page


def test_replay_workflow_runs_the_action_for_real() -> None:
    workflow = yaml.safe_load((ROOT / rp.WORKFLOW).read_text(encoding="utf-8"))
    on = workflow[True]  # YAML 1.1 reads the key `on` as a boolean
    assert "examples/retail_agent/**" in on["pull_request"]["paths"]
    assert workflow["permissions"]["pull-requests"] == "write"
    job = workflow["jobs"]["replay"]
    assert job["name"] == rp.JOB_NAME
    assert job["timeout-minutes"] <= 10
    action = next(s for s in job["steps"] if s.get("uses") == "./")
    assert action["with"]["suite"] == rp.SUITE
    assert action["with"].get("dry-run", "false") == "false"


def test_ci_skips_regression_demo_pull_requests() -> None:
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    ignored = ci[True]["pull_request"]["branches-ignore"]
    assert f"{rp.BRANCH_PREFIX}**" in ignored


def test_rehearsal_fails_the_seeded_regression() -> None:
    result = rp.rehearse(rp.Shell(), rp.DEFAULT_MUTANT, "HEAD")
    assert result["passed"] is False
    failed = {s["name"] for s in result["scenarios"]["failed"]}
    assert failed == {
        "cancel_pending_laptop",
        "modify_address_suite",
        "return_damaged_bicycle",
        "exchange_waterproof_boots",
    }
    assert result["comment"].startswith(rp.marker() + "\n## ❌ Replay: 2/6 scenarios passed")
    assert "Run log and report" not in result["comment"]
