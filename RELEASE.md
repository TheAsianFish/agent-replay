# Releasing agent-replay

Everything a release needs, in order. Steps marked **(once)** are one-time setup.
Nothing here needs a long-lived secret in the repository: PyPI uses trusted
publishing (OIDC), and the container image and GitHub Release use the workflow's
own `GITHUB_TOKEN`.

What a release ships:

| Artefact | Where | Built by |
|---|---|---|
| Wheel and sdist (`pip install agent-replay`), the dashboard bundled in the wheel | PyPI | `.github/workflows/release.yml`, job `publish-pypi` |
| GitHub Release with the wheel, sdist and this version's CHANGELOG section | GitHub Releases | job `github-release` |
| Container image `ghcr.io/theasianfish/agent-replay:<version>` and `:latest` | GitHub Container Registry | job `docker` |
| The GitHub Action (`uses: TheAsianFish/agent-replay@v1`) | this repository, at the `v1` tag | the moving `v1` tag (step 6) |

Every publishing job is guarded by
`github.repository == 'TheAsianFish/agent-replay' && github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')`,
so forks, branch pushes and manual runs never publish, and the build job refuses
a tag that is not a final `X.Y.Z` equal to the package version, or that is not
on `main`. `workflow_dispatch` on `release.yml` runs the build job only: a dry
run of the release.

## Secrets and settings the maintainer adds

| What | Where | Needed for |
|---|---|---|
| PyPI trusted publisher: project `agent-replay`, owner `TheAsianFish`, repository `agent-replay`, workflow `release.yml`, environment `pypi` | pypi.org → Account settings → Publishing → *Add a new pending publisher* | `publish-pypi` (no API token is stored anywhere) |
| GitHub environment `pypi` with yourself as a required reviewer, so every upload waits for a click | repository Settings → Environments → *New environment* → *Required reviewers* | `publish-pypi`; anyone with push access can otherwise publish by pushing a tag, since the workflow's own checks only compare the tag with `pyproject.toml` and `main` |
| Tag ruleset: creation of `v*` tags restricted to maintainers | Settings → Rules → Rulesets → *New tag ruleset*, target `v*`, restrict creations and updates | the same; also keeps the moving `v1` tag (step 6) in maintainers' hands |
| Repository secret `ANTHROPIC_API_KEY` (a capped key) | Settings → Secrets and variables → Actions | the `live` job in `ci.yml` (it skips without the secret) and any key-requiring eval run in Actions |
| Workflow permissions: leave the default (read) | Settings → Actions → General | `release.yml` asks for `id-token`, `contents` and `packages` write per job |

No other secret is used. `GITHUB_TOKEN` covers the PR comment of the Action
(the consumer's workflow grants `pull-requests: write`), the GitHub Release and
the GHCR push.

## 1. Make the repository public (once)

The Action can be used by other repositories, and listed on the Marketplace,
only from a public repository. Settings → General → Danger zone → *Change
visibility*. Before that, check that nothing private is tracked:
`git ls-files | grep -Ei '(^|/)(\.env|\.replay/|.*\.db$)'` must print nothing
(`.gitignore` lists them, so a plain `git add` skips them, but a forced add
would not), and `git grep -nE 'sk-ant-[A-Za-z0-9]'` finds no key.

## 2. Check that main is releasable

1. CI is green on `main` (all jobs, including `package`, `action` and `compose`).
2. Locally, from a clean checkout:

   ```bash
   make check        # lint, mypy --strict, pytest
   make ui-check     # dashboard types, lint, vitest
   make dist         # dashboard build + uv build + install the wheel and sdist in clean venvs
   ```

   `make dist` ends with the clean-venv checks: `replay --version`, a cassette
   read, and `replay serve` answering `/api/health` and serving the bundled
   dashboard at `/`.

## 3. Set the version and the changelog

1. Set `version` in `pyproject.toml` (the package's `__version__` is read from
   the installed metadata, so there is one place to change).
2. In `CHANGELOG.md`, move the entries under `[Unreleased]` into a new
   `## [X.Y.Z] - YYYY-MM-DD` section with today's date, and update the compare
   links at the bottom. The release build refuses a version whose section is
   missing or still says `Unreleased`.
3. `uv lock` (the lockfile records the project version), commit:
   `git commit -am "Release X.Y.Z"`, and push to `main` through a PR as usual.

For 1.0.0 this step is done: `pyproject.toml` and `uv.lock` say `1.0.0`, and
CHANGELOG.md has one dated `[1.0.0]` section holding everything built so far
(no `v1.0.0` tag exists yet, so nothing was split off into a later version).
Its date is the day the release branch was prepared; if the tag lands on a
later day, change that date before tagging, as Keep a Changelog dates a
section by its release day.

## 4. Dry-run the release (optional)

Actions → *Release* → *Run workflow* on `main`. Only the build job runs: tag
check skipped, dashboard build, `uv build`, the clean-venv checks, `twine
check` and the release-notes extraction. Download the `dist` artifact to
inspect the wheel.

## 5. Tag and publish

```bash
git switch main && git pull
git tag -a vX.Y.Z -m "agent-replay X.Y.Z"
git push origin vX.Y.Z
```

The tag starts `release.yml`:

1. `build` checks that the tag is a final `X.Y.Z` (no `-rc1`: the workflow
   publishes no prereleases) equal to the package version and that it points
   into `main`, builds and checks the distributions and extracts the release
   notes;
2. `publish-pypi` uploads to PyPI through trusted publishing (waits for
   approval if the `pypi` environment has a reviewer);
3. `github-release` creates the GitHub Release with the distributions attached;
4. `docker` pushes `ghcr.io/theasianfish/agent-replay:X.Y.Z` and `:latest`.

After the first image push, make the package public once: your profile →
Packages → `agent-replay` → Package settings → *Change visibility*.

Verify: `pip install agent-replay==X.Y.Z` in a fresh venv, `replay --version`.

## 6. Move the Action's major tag

Consumers pin `TheAsianFish/agent-replay@v1`, a tag that moves to the newest
`1.x.y` release, the convention GitHub documents for actions. With
`install: action` (the default) the Action installs the package from the
ref the consumer pinned, so `@v1` and the code it runs always agree.

```bash
git tag -fa v1 vX.Y.Z^{} -m "v1 -> vX.Y.Z"
git push origin v1 --force
```

Only move `v1` for backwards-compatible releases. A breaking change to the
Action's inputs or outputs, or to the suite format, starts `v2`. Consumers who
want an immutable pin use the full tag (`@vX.Y.Z`) or a commit SHA.

## 7. Publish the Action to the Marketplace (optional, once per release)

Needs the public repository, two-factor authentication on the account and an
accepted Marketplace developer agreement. Releases → the release from step 5 →
*Edit* → tick *Publish this Action to the GitHub Marketplace*, choose the
categories *Testing* and *Continuous integration*, and *Update release*.
`action.yml` already carries the name, description and branding the listing
needs.

## 8. Judge calibration data (maintainer work, not a release blocker)

The measured judge-human agreement needs human labels and a key. These are
listed with their costs in RESULTS.md, "To measure":

```bash
make record-golden                     # D1: 120 model-live runs, ~$2.19 estimated
uv run replay label eval/golden/retail-d1 --rubric examples/retail_agent/rubric.yaml --labeller NAME
REPLAY_STATS_IMPL=core make eval-judge # kappa, self-consistency, probes; prints its estimate first
```

Commit the golden set, the labels, the verdict cache and the report, and copy
the numbers into RESULTS.md with their raw output.

## 9. Hosted demo (optional)

A read-only dashboard seeded with the demo data. The API has no authentication,
which is fine here: the image holds only the demo store.

The `demo` stage of the Dockerfile bakes the demo data into a SQLite store at
build time, so the demo is one container with no database service and no
volume:

```bash
docker build --target demo -t agent-replay-demo .
docker run --rm -p 8000:8000 agent-replay-demo     # http://localhost:8000
```

The image is about 285 MB and holds the 480 demo runs; the stage was built and
served locally (`/api/health` healthy on a custom `$PORT`). Neither hosting
route below has been deployed: both need an account.

**Fly.io** (needs `flyctl` and an account; `deploy/fly.toml` asks for one
512 MB shared-CPU machine that stops when idle). From the repository root:

```bash
fly launch --no-deploy --copy-config --config deploy/fly.toml --name agent-replay-demo
fly deploy . --config deploy/fly.toml --dockerfile Dockerfile
```

The repository root is the build context (the Dockerfile copies `dashboard/`,
`agent_replay/` and `examples/` from it), so run both from there.

**Render** (no CLI): New → *Web Service* → this repository → Runtime *Docker*,
Dockerfile path `./Dockerfile`, Docker build target `demo`, health check path
`/api/health`, instance type *Free*. Render sets `PORT`; the demo image listens
on `$PORT` (default 8000).

Then add the URL to the README's dashboard section.
