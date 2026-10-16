"""The GitHub Action's logic (``action.yml``): step outputs and the sticky PR comment.

Stdlib only, so it runs on whichever interpreter the workflow's ``python`` input
names, with or without agent-replay importable there (Python 3.10 or newer).

    summarize  read ``replay ci``'s JSON report and exit code, write the step outputs
    comment    build the PR comment and create or update it (a dry run only writes it)

A failure to comment is a warning, never a failed job: on a pull request from a
fork the token is read-only, and the replay verdict is what gates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

# GitHub rejects comment bodies over 65,536 characters; the body is kept a little
# under that as a safety margin.
COMMENT_LIMIT = 65_536
MAX_BODY = 65_000
PER_PAGE = 100
MAX_PAGES = 100
PR_EVENTS = frozenset({"pull_request", "pull_request_target"})
API_VERSION = "2022-11-28"


class Response(NamedTuple):
    status: int
    body: Any


# (method, url, JSON payload or None) -> response; tests pass a fake.
Http = Callable[[str, str, dict[str, Any] | None], Response]


class CommentError(Exception):
    """The PR comment could not be read or written; reported as a warning."""


def _escape(message: str) -> str:
    """Escape ``message`` as workflow-command data (one annotation, one line)."""
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def warning(message: str) -> None:
    print(f"::warning::{_escape(message)}")


def error(message: str) -> None:
    print(f"::error::{_escape(message)}")


def write_outputs(values: Mapping[str, str], env: Mapping[str, str]) -> None:
    """Append ``values`` to ``$GITHUB_OUTPUT`` (printed instead when it is unset)."""
    lines = []
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            delimiter = f"EOF_{secrets.token_hex(8)}"
            lines += [f"{key}<<{delimiter}", value, delimiter]
        else:
            lines.append(f"{key}={value}")
    text = "\n".join(lines) + "\n"
    if path := env.get("GITHUB_OUTPUT"):
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


def _load_report(path: Path) -> dict[str, Any] | None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        warning(f"cannot read the replay report {path}: {exc}")
        return None
    if not isinstance(report, dict) or not isinstance(report.get("scenarios"), list):
        warning(f"the replay report {path} has no scenarios list; is it a `replay ci` report?")
        return None
    return report


def summarize(report_json: Path, report_markdown: Path, exit_code: int) -> dict[str, str]:
    """The step outputs for one ``replay ci`` run; ``passed`` follows the exit code."""
    report = _load_report(report_json)
    scenarios: list[Any] = report["scenarios"] if report is not None else []
    failed = sum(1 for s in scenarios if not (isinstance(s, dict) and s.get("passed") is True))
    if report is None and exit_code != 0:
        error(
            f"replay ci exited {exit_code} without a report: the suite could not be loaded "
            "or the command did not start; its error is in the log above"
        )
    elif report is None:
        warning("replay ci exited 0 but wrote no readable report")
    else:
        print(f"replay ci: {len(scenarios) - failed}/{len(scenarios)} scenario(s) passed")
    return {
        "passed": "true" if exit_code == 0 else "false",
        "scenarios": str(len(scenarios)),
        "failed": str(failed),
        "report-json": str(report_json) if report is not None else "",
        "report-markdown": str(report_markdown) if report_markdown.is_file() else "",
    }


def marker(suite: str) -> str:
    """The hidden first line that identifies this suite's comment on a pull request."""
    clean = re.sub(r"-(?=-)", "-\u2060", " ".join(suite.split()))  # never close the comment
    return f"<!-- agent-replay: {clean} -->"


def run_url(env: Mapping[str, str]) -> str | None:
    server = env.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    repo, run_id = env.get("GITHUB_REPOSITORY"), env.get("GITHUB_RUN_ID")
    return f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else None


def _truncate(markdown: str, budget: int) -> str:
    """Cut ``markdown`` at a line boundary to ``budget`` characters, closing an open fence."""
    note = "\n\n_Summary truncated to fit a PR comment; the full report is in the run's artifact._"
    room = budget - len(note) - len("\n  ```")
    cut = markdown[: max(room, 0)]
    if "\n" in cut:
        cut = cut[: cut.rindex("\n")]
    if sum(1 for line in cut.splitlines() if line.lstrip().startswith("```")) % 2:
        cut += "\n  ```"
    return cut + note


def build_body(
    suite: str, markdown: str | None, exit_code: int, url: str | None, *, limit: int = MAX_BODY
) -> str:
    """The comment: marker, ``replay ci``'s Markdown summary and a link to the run."""
    head = marker(suite) + "\n"
    if markdown is None or not markdown.strip():
        markdown = (
            "## ⚠️ Replay: the suite could not run\n\n"
            f"`replay ci` exited {exit_code} on `{suite}` without a summary; "
            "the error is in the run log.\n"
        )
    foot = f"\n---\n[Run log and report]({url})\n" if url else ""
    budget = limit - len(head) - len(foot)
    if len(markdown) > budget:
        markdown = _truncate(markdown, budget) + "\n"
    return head + markdown + foot


def pr_number(env: Mapping[str, str]) -> int | None:
    """The pull request number from ``$GITHUB_EVENT_PATH``, or None off pull request events."""
    if env.get("GITHUB_EVENT_NAME") not in PR_EVENTS or not env.get("GITHUB_EVENT_PATH"):
        return None
    try:
        event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CommentError(f"cannot read the event payload: {exc}") from exc
    number = (event.get("pull_request") or {}).get("number", event.get("number"))
    if not isinstance(number, int):
        raise CommentError("the event payload has no pull request number")
    return number


def _check(response: Response, what: str) -> Any:
    if 200 <= response.status < 300:
        return response.body
    detail = response.body.get("message") if isinstance(response.body, dict) else None
    hint = ""
    if response.status in (401, 403, 404):
        hint = (
            "; the token cannot write to this pull request (a fork's pull_request run gets "
            "a read-only token): grant `pull-requests: write`, or set `comment: false`"
        )
    raise CommentError(f"{what} failed: HTTP {response.status} {detail or ''}".rstrip() + hint)


def find_comments(http: Http, api: str, repo: str, number: int, mark: str) -> list[int]:
    """Ids of the pull request's comments that start with ``mark``, bots' first.

    Anyone can post a comment that starts with the marker, and the token can edit
    only its own, so every match is a candidate: a bot (``github-actions[bot]``
    for the default token) is the likely author, then the rest in posting order.
    """
    bots: list[int] = []
    others: list[int] = []
    for page in range(1, MAX_PAGES + 1):
        url = f"{api}/repos/{repo}/issues/{number}/comments?per_page={PER_PAGE}&page={page}"
        comments = _check(http("GET", url, None), "listing the pull request's comments")
        if not isinstance(comments, list):
            raise CommentError("listing the pull request's comments returned no list")
        for comment in comments:
            if str(comment.get("body", "")).startswith(mark):
                user = comment.get("user") or {}
                (bots if user.get("type") == "Bot" else others).append(int(comment["id"]))
        if len(comments) < PER_PAGE:
            break
    return bots + others


def upsert(http: Http, api: str, repo: str, number: int, mark: str, body: str) -> str:
    """Update the suite's comment on the pull request, or create it; returns its URL.

    A candidate the token may not edit (403 or 404: someone else wrote it) is
    skipped; when none can be edited, a new comment is created.
    """
    for existing in find_comments(http, api, repo, number, mark):
        url = f"{api}/repos/{repo}/issues/comments/{existing}"
        response = http("PATCH", url, {"body": body})
        if response.status in (403, 404):
            continue
        result = _check(response, "updating the comment")
        break
    else:
        url = f"{api}/repos/{repo}/issues/{number}/comments"
        result = _check(http("POST", url, {"body": body}), "creating the comment")
    return str(result.get("html_url", "")) if isinstance(result, dict) else ""


def urllib_http(token: str, *, timeout: float = 30.0) -> Http:
    """The real HTTP layer: GitHub's REST API over urllib, authenticated with ``token``."""

    def request(method: str, url: str, payload: dict[str, Any] | None) -> Response:
        if not url.startswith("https://"):
            raise CommentError(f"refusing a non-HTTPS API URL: {url}")
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)  # noqa: S310 - https only
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", API_VERSION)
        req.add_header("User-Agent", "agent-replay-action")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return Response(resp.status, _json_or_none(resp.read()))
        except urllib.error.HTTPError as exc:
            return Response(exc.code, _json_or_none(exc.read()))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise CommentError(f"cannot reach the GitHub API: {exc}") from exc

    return request


def _json_or_none(raw: bytes) -> Any:
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def comment(
    *,
    suite: str,
    markdown_path: Path,
    exit_code: int,
    body_path: Path,
    post: bool,
    dry_run: bool,
    env: Mapping[str, str],
    http: Http | None = None,
) -> dict[str, str]:
    """Write the comment body to ``body_path`` and, unless told not to, post it."""
    try:
        markdown: str | None = markdown_path.read_text(encoding="utf-8")
    except OSError:
        markdown = None
    body = build_body(suite, markdown, exit_code, run_url(env))
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text(body, encoding="utf-8")
    outputs = {"comment-body": str(body_path), "comment-url": ""}
    if not post:
        print("PR comment disabled (comment: false)")
    else:
        outputs["comment-url"] = _post(suite, body, dry_run=dry_run, env=env, http=http)
    return outputs


def _post(
    suite: str, body: str, *, dry_run: bool, env: Mapping[str, str], http: Http | None
) -> str:
    """Upsert ``body`` on the event's pull request; the comment's URL, or "" when none."""
    try:
        number = pr_number(env)
    except CommentError as exc:
        warning(f"agent-replay did not comment: {exc}")
        if not dry_run:
            return ""
        number = None
    if dry_run:
        _print_dry_run(body, number)
        return ""
    if number is None:
        print(f"not a pull request event ({env.get('GITHUB_EVENT_NAME', 'none')}); no comment")
        return ""
    token, repo = env.get("GITHUB_TOKEN", ""), env.get("GITHUB_REPOSITORY", "")
    if not token or not repo:
        warning("agent-replay did not comment: no github-token or GITHUB_REPOSITORY")
        return ""
    api = env.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    try:
        url = upsert(http or urllib_http(token), api, repo, number, marker(suite), body)
    except (CommentError, OSError, ValueError, KeyError, TypeError) as exc:
        warning(f"agent-replay did not comment on #{number}: {exc}")
        return ""
    print(f"PR comment: {url}")
    return url


def _print_dry_run(body: str, number: int | None) -> None:
    # The body holds agent output; stop-commands keeps any "::" line in it inert.
    stop = secrets.token_hex(16)
    target = f"for #{number}" if number is not None else "(no pull request to post to)"
    print(f"::group::PR comment {target}: dry run, not posted")
    print(f"::stop-commands::{stop}")
    print(body)
    print(f"::{stop}::")
    print("::endgroup::")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="replay_action", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summarize", help="write the step outputs for a replay ci run")
    s.add_argument("--report-json", type=Path, required=True)
    s.add_argument("--report-markdown", type=Path, required=True)
    s.add_argument("--exit-code", type=int, required=True)
    c = sub.add_parser("comment", help="write the PR comment and create or update it")
    c.add_argument("--suite", required=True)
    c.add_argument("--report-markdown", type=Path, required=True)
    c.add_argument("--exit-code", type=int, required=True)
    c.add_argument("--body", type=Path, required=True, help="where to write the comment body")
    c.add_argument("--post", choices=["true", "false"], default="true")
    c.add_argument("--dry-run", choices=["true", "false"], default="false")
    return parser


def main(
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    http: Http | None = None,
) -> int:
    args = _parser().parse_args(argv)
    env = os.environ if env is None else env
    if args.command == "summarize":
        outputs = summarize(args.report_json, args.report_markdown, args.exit_code)
    else:
        outputs = comment(
            suite=args.suite,
            markdown_path=args.report_markdown,
            exit_code=args.exit_code,
            body_path=args.body,
            post=args.post == "true",
            dry_run=args.dry_run == "true",
            env=env,
            http=http,
        )
    write_outputs(outputs, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
