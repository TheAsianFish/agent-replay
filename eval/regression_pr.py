"""Question 3: does the GitHub Action stop a bad pull request in practice?

``make regression-pr`` opens a real pull request on this repository that points
the retail suite at a seeded mutant, lets ``.github/workflows/replay.yml`` run
the Action for real (``dry-run`` off), and captures the evidence: the PR, the
failing check run and its wall time, the scenarios that failed (from the run's
report artifact) and the sticky comment's permalink, body and rendered HTML.
Then it closes the PR and deletes its branches; the comment stays on the closed
PR. Everything is written to ``eval/results/regression_pr/`` plus
``docs/screenshots/pr-comment.png``, and the RESULTS.md section is printed.

The PR's base is a fresh copy of the source ref (``regression-demo/base-*``),
which ci.yml ignores, so the full CI does not spend minutes on a PR built to
fail; only the replay check runs.

It needs ``gh`` logged in as someone who can push to the repository (the
maintainer's own login; no new secret) and the source ref (default
``origin/main``) to contain ``.github/workflows/replay.yml``.

``--rehearse`` does everything that needs no GitHub: the same suite edit in a
throwaway worktree, the same ``replay ci`` command the Action runs, and the
comment body the Action would post (built by ``action/replay_action.py``).

Usage::

    uv run python -m eval.regression_pr              # the real PR (needs gh auth)
    uv run python -m eval.regression_pr --rehearse   # offline, no GitHub
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUITE = "examples/retail_agent/replay.toml"
WORKFLOW = ".github/workflows/replay.yml"
JOB_NAME = "Replay the retail suite"
DEFAULT_AGENT = 'agent = "examples.retail_agent.agent:run"'
DEFAULT_MUTANT = "wrong_write_argument"
BRANCH_PREFIX = "regression-demo/"
OUT = ROOT / "eval" / "results" / "regression_pr"
SCREENSHOT = ROOT / "docs" / "screenshots" / "pr-comment.png"
WORKFLOW_NAME = "Agent replay"
COMMENT_AUTHOR = "github-actions[bot]"


class RegressionPrError(Exception):
    """A step failed; the message names the fix."""


@dataclass
class Shell:
    """Runs a command and returns its stdout; tests replace :meth:`run`."""

    env: Mapping[str, str] | None = None
    log: list[list[str]] = field(default_factory=list)

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> str:
        self.log.append(list(argv))
        proc = subprocess.run(  # noqa: S603 - argv is built in this module, never a shell
            list(argv),
            cwd=cwd,
            env=env if env is not None else self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and proc.returncode != 0:
            raise RegressionPrError(
                f"`{' '.join(argv)}` exited {proc.returncode}: {proc.stderr.strip()}"
            )
        return proc.stdout


def seed_regression(suite_text: str, mutant: str) -> str:
    """The suite with its default agent swapped for ``examples.retail_agent.mutants:<mutant>``."""
    if suite_text.count(DEFAULT_AGENT) != 1:
        raise RegressionPrError(
            f"{SUITE} must set `{DEFAULT_AGENT}` exactly once in [defaults]; "
            "update DEFAULT_AGENT in eval/regression_pr.py to match the suite"
        )
    return suite_text.replace(DEFAULT_AGENT, f'agent = "examples.retail_agent.mutants:{mutant}"')


def check_mutant(mutant: str) -> None:
    """Refuse a mutant name the suite could not import: a broken suite is not a regression."""
    module = importlib.import_module("examples.retail_agent.mutants")
    if not callable(getattr(module, mutant, None)):
        # Each mutant is a bound ``run`` of a RetailAgent subclass.
        names = sorted(n for n, v in vars(module).items() if inspect.ismethod(v))
        raise RegressionPrError(
            f"no mutant {mutant!r} in examples/retail_agent/mutants.py: {names}"
        )


def check_regression(report: Mapping[str, Any]) -> None:
    """The report shows a caught regression: some scenarios failed on checks, none broke.

    ``replay ci`` exits 2 and fails every scenario when the suite cannot run (a bad
    agent path, a missing cassette); that also fails the check, but it is not the
    Action stopping a regression, so it is refused as evidence.
    """
    scenarios = list(report.get("scenarios", []))
    failed = [s for s in scenarios if s.get("passed") is not True]
    broken = [str(s.get("name")) for s in failed if s.get("error") or not s.get("ran", True)]
    if broken:
        raise RegressionPrError(
            f"scenarios could not run (a broken suite, not a regression): {broken}"
        )
    if report.get("passed") is not False or not failed:
        raise RegressionPrError("the report has no failed scenario: the regression was not caught")
    unexplained = [
        str(s.get("name"))
        for s in failed
        if not any(not c.get("passed") for c in s.get("checks", []))
    ]
    if unexplained:
        raise RegressionPrError(f"scenarios failed without a failed check: {unexplained}")


def origin_repo(url: str) -> str:
    """``owner/name`` from origin's URL (https or ssh), where the branches are pushed."""
    found = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url.strip())
    if not found:
        raise RegressionPrError(f"origin ({url.strip()}) is not a GitHub repository")
    return found.group(1)


def marker(suite: str = SUITE) -> str:
    """The sticky comment's first line (see ``action/replay_action.py``)."""
    return f"<!-- agent-replay: {suite} -->"


def scenario_outcomes(report: Mapping[str, Any]) -> dict[str, Any]:
    """Which scenarios failed and on which checks, from a ``replay ci`` JSON report."""
    failed: list[dict[str, Any]] = []
    passed: list[str] = []
    for scenario in report.get("scenarios", []):
        if scenario.get("passed") is True:
            passed.append(str(scenario["name"]))
            continue
        checks = [
            str(c.get("check", "?")) for c in scenario.get("checks", []) if not c.get("passed")
        ]
        failed.append({"name": str(scenario["name"]), "failed_checks": checks})
    return {"failed": failed, "passed": passed}


def _parse_time(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def seconds_between(start: str, end: str) -> float:
    return (_parse_time(end) - _parse_time(start)).total_seconds()


def pick_job(run: Mapping[str, Any], name: str = JOB_NAME) -> dict[str, Any]:
    for job in run.get("jobs", []):
        if job.get("name") == name:
            return dict(job)
    names = [str(j.get("name")) for j in run.get("jobs", [])]
    raise RegressionPrError(
        f"run has no job named {name!r} (jobs: {names}); was {WORKFLOW} edited?"
    )


def pick_comment(comments: Sequence[Mapping[str, Any]], suite: str = SUITE) -> dict[str, Any]:
    """The Action's sticky comment: posted by the Actions bot and starting with the marker."""
    for comment in comments:
        user = comment.get("user") or {}
        body = str(comment.get("body", ""))
        if user.get("login") == COMMENT_AUTHOR and body.startswith(marker(suite)):
            return dict(comment)
    raise RegressionPrError(
        "the pull request has no agent-replay comment from github-actions[bot]; check the "
        "run's 'Comment on the pull request' step (the workflow needs pull-requests: write)"
    )


def results_markdown(evidence: Mapping[str, Any]) -> str:
    """The RESULTS.md section, from the captured evidence only."""
    failed = evidence["scenarios"]["failed"]
    passed = evidence["scenarios"]["passed"]
    lines = [
        f"- Pull request: {evidence['pr_url']} (seeded mutant `{evidence['mutant']}`, "
        f"head `{evidence['head_sha'][:12]}`, closed after capture)",
        f"- Failing check run: {evidence['check_url']} (conclusion `{evidence['conclusion']}`)",
        f"- Sticky comment: {evidence['comment_url']}",
        f"- Check wall time: {evidence['job_wall_seconds']:.0f} s for the job "
        f"({evidence['run_wall_seconds']:.0f} s for the whole run, queueing included)",
        f"- Scenarios failed: {len(failed)} of {len(failed) + len(passed)}",
    ]
    for scenario in failed:
        lines.append(f"  - `{scenario['name']}`: {', '.join(scenario['failed_checks'])}")
    if passed:
        lines.append(f"- Passed: {', '.join(f'`{name}`' for name in passed)}")
    return "\n".join(lines) + "\n"


COMMENT_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
body {{ margin: 0; padding: 24px; background: #fff; font: 14px/1.5 -apple-system,
  BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2328; }}
.box {{ max-width: 920px; border: 1px solid #d1d9e0; border-radius: 6px; }}
.head {{ background: #f6f8fa; border-bottom: 1px solid #d1d9e0; padding: 8px 16px;
  border-radius: 6px 6px 0 0; color: #59636e; }}
.head b {{ color: #1f2328; }}
.body {{ padding: 16px; }}
h2 {{ font-size: 1.5em; border-bottom: 1px solid #d1d9e0; padding-bottom: .3em; margin-top: 0; }}
h3 {{ font-size: 1.25em; }}
table {{ border-collapse: collapse; margin: 0 0 16px; }}
th, td {{ border: 1px solid #d1d9e0; padding: 6px 13px; }}
tr:nth-child(2n) {{ background: #f6f8fa; }}
code {{ background: #eff1f3; padding: .2em .4em; border-radius: 6px; font-size: 85%;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
a {{ color: #0969da; text-decoration: none; }}
hr {{ border: 0; border-top: 1px solid #d1d9e0; }}
</style></head><body><div class="box">
<div class="head"><b>{author}</b> commented on <a href="{url}">{pr}</a> &middot; {created}</div>
<div class="body">{html}</div></div></body></html>
"""


def comment_page(comment: Mapping[str, Any], pr_url: str) -> str:
    """The comment's GitHub-rendered HTML (``body_html``) in a page for the screenshot."""
    return COMMENT_PAGE.format(
        author=COMMENT_AUTHOR,
        url=comment["html_url"],
        pr="#" + pr_url.rstrip("/").rsplit("/", 1)[-1],
        created=comment.get("created_at", ""),
        html=comment["body_html"],
    )


@dataclass
class Demo:
    shell: Shell
    mutant: str = DEFAULT_MUTANT
    source: str = "origin/main"
    timeout_s: float = 900.0
    poll_s: float = 10.0
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    root: Path = ROOT
    out: Path = OUT
    screenshot: Path = SCREENSHOT

    def gh_json(self, *args: str) -> Any:
        return json.loads(self.shell.run(["gh", *args]))

    # -- preflight -------------------------------------------------------------------

    def preflight(self) -> str:
        try:
            self.shell.run(["gh", "auth", "status"])
        except (RegressionPrError, FileNotFoundError) as exc:
            raise RegressionPrError(
                "gh is not installed or not logged in; install the GitHub CLI and run "
                f"`gh auth login` as a maintainer of the repository ({exc})"
            ) from exc
        check_mutant(self.mutant)
        self.shell.run(["git", "fetch", "--quiet", "origin"], cwd=self.root)
        try:
            self.shell.run(["git", "cat-file", "-e", f"{self.source}:{WORKFLOW}"], cwd=self.root)
            ci = self.shell.run(["git", "show", f"{self.source}:.github/workflows/ci.yml"])
        except RegressionPrError as exc:
            raise RegressionPrError(
                f"{self.source} has no {WORKFLOW}: merge the change that adds it first, or "
                "pass --from <a pushed branch that has it>"
            ) from exc
        if f"{BRANCH_PREFIX}**" not in ci:
            raise RegressionPrError(
                f"ci.yml at {self.source} does not ignore {BRANCH_PREFIX}** bases, so the full "
                "CI would run on the demo PR; use a source that has that change"
            )
        # The branches go to origin, so the PR is opened there (not at gh's default repo).
        return origin_repo(self.shell.run(["git", "remote", "get-url", "origin"], cwd=self.root))

    # -- the pull request ------------------------------------------------------------

    def open_pr(self, repo: str, worktree: Path, base: str, head: str) -> tuple[str, str]:
        """Push the base and the seeded head, open the PR; its URL and the head commit."""
        sha = self.shell.run(["git", "rev-parse", self.source], cwd=self.root).strip()
        self.shell.run(["git", "worktree", "add", "-b", head, str(worktree), sha], cwd=self.root)
        suite = worktree / SUITE
        suite.write_text(
            seed_regression(suite.read_text(encoding="utf-8"), self.mutant), encoding="utf-8"
        )
        self.shell.run(["git", "add", SUITE], cwd=worktree)
        self.shell.run(
            [
                "git",
                "commit",
                "--quiet",
                "-m",
                f"Point the retail suite at the {self.mutant} mutant",
            ],
            cwd=worktree,
        )
        head_sha = self.shell.run(["git", "rev-parse", "HEAD"], cwd=worktree).strip()
        self.shell.run(
            ["git", "push", "--quiet", "origin", f"{sha}:refs/heads/{base}"], cwd=self.root
        )
        self.shell.run(
            ["git", "push", "--quiet", "origin", f"{head}:refs/heads/{head}"], cwd=worktree
        )
        body = (
            "A seeded regression, opened by `make regression-pr` to show the replay check "
            f"failing a pull request: the retail suite's agent becomes the `{self.mutant}` "
            "mutant from `examples/retail_agent/mutants.py`. The base is a copy of "
            f"`{self.source}` at `{sha[:12]}` so only the replay workflow runs. "
            "This PR is closed once the evidence is captured; do not merge it.\n"
        )
        url = (
            self.shell.run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    repo,
                    "--base",
                    base,
                    "--head",
                    head,
                    "--title",
                    f"Seeded regression: {self.mutant} (do not merge)",
                    "--body",
                    body,
                ],
                cwd=worktree,
            )
            .strip()
            .splitlines()[-1]
        )
        return url, head_sha

    def wait_for_run(self, repo: str, head: str, head_sha: str) -> dict[str, Any]:
        # Filtered here, not with --workflow: gh resolves a file name through the default
        # branch and errors out while the workflow has never run there.
        deadline = self.timeout_s
        waited = 0.0
        while True:
            runs = self.gh_json(
                "run",
                "list",
                "--repo",
                repo,
                "--branch",
                head,
                "--commit",
                head_sha,
                "--event",
                "pull_request",
                "--json",
                "databaseId,status,conclusion,url,headSha,workflowName",
                "--limit",
                "10",
            )
            runs = [r for r in runs if r.get("workflowName") == WORKFLOW_NAME]
            done = [r for r in runs if r.get("status") == "completed"]
            if done:
                return dict(done[0])
            if waited >= deadline:
                state = runs[0]["status"] if runs else "not started"
                raise RegressionPrError(
                    f"the replay run on {head} is {state} after {deadline:.0f} s; watch it with "
                    f"`gh run list --branch {head}` and raise --timeout"
                )
            self.sleep(self.poll_s)
            waited += self.poll_s

    def collect(self, repo: str, pr_url: str, run_id: int, scratch: Path) -> dict[str, Any]:
        run = self.gh_json(
            "run",
            "view",
            str(run_id),
            "--repo",
            repo,
            "--json",
            "jobs,url,conclusion,createdAt,updatedAt,headSha",
        )
        job = pick_job(run)
        artifacts = scratch / "artifacts"
        self.shell.run(["gh", "run", "download", str(run_id), "--repo", repo, "-D", str(artifacts)])
        reports = sorted(artifacts.rglob("report.json"))
        if not reports:
            raise RegressionPrError(f"run {run_id} uploaded no report.json; see its log")
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        number = pr_url.rstrip("/").rsplit("/", 1)[-1]
        comments = self.gh_json("api", f"repos/{repo}/issues/{number}/comments")
        found = pick_comment(comments)
        rendered = self.gh_json(
            "api",
            "-H",
            "Accept: application/vnd.github.full+json",
            f"repos/{repo}/issues/comments/{found['id']}",
        )
        found["body_html"] = rendered["body_html"]
        if str(job.get("conclusion")) != "failure":
            raise RegressionPrError(
                f"the replay check concluded {job.get('conclusion')!r}, not 'failure': the "
                f"Action did not stop the seeded regression ({job.get('url')})"
            )
        check_regression(report)
        staged = scratch / "evidence"
        staged.mkdir()
        (staged / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (staged / "comment.md").write_text(str(found["body"]), encoding="utf-8")
        page = scratch / "comment.html"
        page.write_text(comment_page(found, pr_url), encoding="utf-8")
        evidence = {
            "question": "Does the GitHub Action stop a bad PR in practice?",
            "pr_url": pr_url,
            "mutant": self.mutant,
            "source": self.source,
            "head_sha": str(run["headSha"]),
            "run_url": str(run["url"]),
            "check_url": str(job["url"]),
            "conclusion": str(job["conclusion"]),
            "job_started_at": str(job["startedAt"]),
            "job_completed_at": str(job["completedAt"]),
            "job_wall_seconds": seconds_between(job["startedAt"], job["completedAt"]),
            "run_wall_seconds": seconds_between(run["createdAt"], run["updatedAt"]),
            "replay_ci_duration_ms": report.get("duration_ms"),
            "scenarios": scenario_outcomes(report),
            "comment_url": str(found["html_url"]),
            "comment_created_at": str(found.get("created_at", "")),
            "captured_at": self.now().isoformat(timespec="seconds"),
        }
        (staged / "evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        shot = scratch / "pr-comment.png"
        self.take_screenshot(page, shot)
        # Only now, with everything captured and checked, replace the committed evidence.
        self.out.mkdir(parents=True, exist_ok=True)
        for item in staged.iterdir():
            shutil.copyfile(item, self.out / item.name)
        self.screenshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shot, self.screenshot)
        return evidence

    def take_screenshot(self, page: Path, target: Path) -> None:
        self.shell.run(
            [
                "npm",
                "--prefix",
                str(self.root / "dashboard"),
                "exec",
                "--",
                "playwright",
                "screenshot",
                "--full-page",
                "--viewport-size=968,600",
                page.as_uri(),
                str(target),
            ],
            cwd=self.root,
        )

    def close(self, repo: str, pr_url: str, base: str) -> None:
        """Close the PR and delete both remote branches; the comment stays on the closed PR."""
        self.shell.run(
            [
                "gh",
                "pr",
                "close",
                pr_url,
                "--repo",
                repo,
                "--delete-branch",
                "--comment",
                "Evidence captured (check run, comment, report); closing the seeded regression.",
            ],
            cwd=self.root,
        )
        self.shell.run(["git", "push", "--quiet", "origin", "--delete", base], cwd=self.root)

    def forget_worktree(self, worktree: Path, head: str) -> None:
        """Drop the throwaway worktree and its local branch, whatever happened."""
        self.shell.run(
            ["git", "worktree", "remove", "--force", str(worktree)], cwd=self.root, check=False
        )
        self.shell.run(["git", "worktree", "prune"], cwd=self.root, check=False)
        # gh pr close --repo deletes only the remote branch.
        self.shell.run(["git", "branch", "-D", head], cwd=self.root, check=False)

    def run(self) -> dict[str, Any]:
        repo = self.preflight()
        stamp = self.now().strftime("%Y%m%d-%H%M%S")
        base, head = f"{BRANCH_PREFIX}base-{stamp}", f"{BRANCH_PREFIX}{stamp}"
        pr_url = ""
        with tempfile.TemporaryDirectory(prefix="regression-pr-") as tmp:
            scratch = Path(tmp)
            worktree = scratch / "worktree"
            try:
                pr_url, head_sha = self.open_pr(repo, worktree, base, head)
                print(f"opened {pr_url}; waiting for the replay check", file=sys.stderr)
                run = self.wait_for_run(repo, head, head_sha)
                evidence = self.collect(repo, pr_url, int(run["databaseId"]), scratch)
            except BaseException:
                close = f"gh pr close {pr_url} --repo {repo} --delete-branch; " if pr_url else ""
                print(
                    f"regression-pr stopped; nothing was written to {self.out}. Left for "
                    f"inspection: {pr_url or 'no PR'}. Clean up with: {close}"
                    f"git push origin --delete {base} {head}",
                    file=sys.stderr,
                )
                raise
            finally:
                self.forget_worktree(worktree, head)
            self.close(repo, pr_url, base)
        return evidence


def rehearse(shell: Shell, mutant: str, source: str, root: Path = ROOT) -> dict[str, Any]:
    """The same edit and the same ``replay ci`` command in a throwaway worktree; no GitHub."""
    check_mutant(mutant)
    with tempfile.TemporaryDirectory(prefix="regression-pr-") as tmp:
        worktree = Path(tmp) / "worktree"
        shell.run(["git", "worktree", "add", "--detach", str(worktree), source], cwd=root)
        try:
            suite = worktree / SUITE
            suite.write_text(
                seed_regression(suite.read_text(encoding="utf-8"), mutant), encoding="utf-8"
            )
            report_json, report_md = Path(tmp) / "report.json", Path(tmp) / "report.md"
            started = time.perf_counter()
            # What action.yml runs: `<python> -m agent_replay ci <suite> --json --markdown <args>`.
            shell.run(
                [
                    sys.executable,
                    "-m",
                    "agent_replay",
                    "ci",
                    SUITE,
                    "--json",
                    str(report_json),
                    "--markdown",
                    str(report_md),
                    "--no-store",
                ],
                cwd=worktree,
                check=False,
            )
            wall = time.perf_counter() - started
            body = Path(tmp) / "comment.md"
            shell.run(
                [
                    sys.executable,
                    "-I",
                    str(worktree / "action" / "replay_action.py"),
                    "comment",
                    "--suite",
                    SUITE,
                    "--report-markdown",
                    str(report_md),
                    "--exit-code",
                    "1",
                    "--body",
                    str(body),
                    "--post",
                    "false",
                ],
                cwd=worktree,
                # Off GitHub Actions: no run link from whatever runner this happens on.
                env={k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")},
            )
            report = json.loads(report_json.read_text(encoding="utf-8"))
            check_regression(report)
            return {
                "mutant": mutant,
                "source": source,
                "replay_ci_wall_seconds": round(wall, 2),
                "passed": report["passed"],
                "scenarios": scenario_outcomes(report),
                "comment": body.read_text(encoding="utf-8"),
            }
        finally:
            shell.run(
                ["git", "worktree", "remove", "--force", str(worktree)], cwd=root, check=False
            )
            shell.run(["git", "worktree", "prune"], cwd=root, check=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mutant", default=DEFAULT_MUTANT, help="a name in examples/retail_agent/mutants.py"
    )
    parser.add_argument(
        "--from",
        dest="source",
        default=None,
        help="ref the PR starts from (default origin/main; HEAD when rehearsing)",
    )
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="seconds to wait for the check"
    )
    parser.add_argument("--rehearse", action="store_true", help="offline: no push, no PR")
    args = parser.parse_args(argv)
    shell = Shell()
    try:
        if args.rehearse:
            result = rehearse(shell, args.mutant, args.source or "HEAD")
            print(result.pop("comment"))
            print(json.dumps(result, indent=2))
            return 0
        evidence = Demo(
            shell, mutant=args.mutant, source=args.source or "origin/main", timeout_s=args.timeout
        ).run()
    except RegressionPrError as exc:
        print(f"regression-pr: {exc}", file=sys.stderr)
        return 2
    print(results_markdown(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
