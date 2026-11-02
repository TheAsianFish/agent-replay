#!/usr/bin/env bash
# Install the built distributions into fresh virtual environments, away from the
# repository, and check that they work: the wheel with the server extra (CLI, SDK
# import, a cassette shown and replayed, the migrations, `replay serve` answering
# /api/health and /), then the sdist (builds and reports its version).
#
#   scripts/check_wheel.sh [--no-dashboard] [DIST_DIR]    (default DIST_DIR: dist)
#
# The wheel must bundle the dashboard (a build from a checkout without `make ui` is
# valid to pip but serves a 404 at /); --no-dashboard allows such a wheel.
# PYTHON picks the interpreter (default python3.12; when that is not on PATH, the
# CPython 3.12 uv manages, `uv python find --system 3.12`). Needs curl; pip fetches
# the dependencies from the package index.
set -euo pipefail

expect_dashboard=1
dist=""
for arg in "$@"; do
    case "$arg" in
        --expect-dashboard) expect_dashboard=1 ;;
        --no-dashboard) expect_dashboard=0 ;;
        -h | --help) sed -n '2,10p' "$0"; exit 0 ;;
        -*) echo "check_wheel: unknown option $arg" >&2; exit 2 ;;
        *) dist="$arg" ;;
    esac
done
repo=$(cd "$(dirname "$0")/.." && pwd)
dist=$(cd "${dist:-$repo/dist}" && pwd)
python=${PYTHON:-python3.12}

fail() { echo "check_wheel: FAIL: $*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

if [ -z "${PYTHON:-}" ] && ! command -v "$python" > /dev/null 2>&1; then
    # No python3.12 on PATH: use the one uv manages (outside the project's .venv).
    python=$(uv python find --system 3.12 2> /dev/null) \
        || fail "python3.12 is not on PATH and uv has no CPython 3.12; run \`uv python install 3.12\` or set PYTHON"
    echo "python: $python (python3.12 is not on PATH)"
fi

shopt -s nullglob
wheels=("$dist"/agent_replay-*.whl)
sdists=("$dist"/agent_replay-*.tar.gz)
shopt -u nullglob
[ "${#wheels[@]}" -eq 1 ] || fail "expected one agent_replay wheel in $dist, found ${#wheels[@]}; run \`uv build\`"
[ "${#sdists[@]}" -eq 1 ] || fail "expected one agent_replay sdist in $dist, found ${#sdists[@]}; run \`uv build\`"
wheel=${wheels[0]}
sdist=${sdists[0]}
version=$(basename "$wheel" | cut -d- -f2)
echo "wheel: $wheel ($(du -k "$wheel" | cut -f1) KiB)"
echo "sdist: $sdist ($(du -k "$sdist" | cut -f1) KiB)"
echo "version: $version"

tmp=$(mktemp -d)
server=""
cleanup() {
    if [ -n "$server" ]; then
        kill "$server" 2>/dev/null || true
        wait "$server" 2>/dev/null || true
    fi
    rm -rf "$tmp"
}
trap cleanup EXIT
# Run from the temporary directory so nothing in the repository is importable.
cd "$tmp"
unset PYTHONPATH VIRTUAL_ENV

step "wheel: fresh venv, pip install '$(basename "$wheel")[server]'"
"$python" -m venv "$tmp/venv"
venv="$tmp/venv/bin"
"$venv/python" -m pip install --quiet --disable-pip-version-check "${wheel}[server]"

step "replay --version"
out=$("$venv/replay" --version)
echo "$out"
[ "$out" = "agent-replay $version" ] || fail "replay --version printed '$out', expected 'agent-replay $version'"

step "import agent_replay from the venv"
"$venv/python" -I - "$version" "$expect_dashboard" <<'PY'
import sys
from importlib.resources import files
from pathlib import Path

import agent_replay

version, expect_dashboard = sys.argv[1], sys.argv[2] == "1"
location = Path(agent_replay.__file__).parent
assert "site-packages" in location.parts, f"imported from {location}, not the venv"
assert agent_replay.__version__ == version, (agent_replay.__version__, version)
package = files("agent_replay")
assert package.joinpath("py.typed").is_file(), "py.typed missing"
migrations = package.joinpath("migrations")
assert migrations.joinpath("env.py").is_file(), "migrations/env.py missing"
assert migrations.joinpath("script.py.mako").is_file(), "migrations/script.py.mako missing"
revisions = sorted(
    p.name for p in migrations.joinpath("versions").iterdir() if p.name[:4].isdigit()
)
assert revisions, "no Alembic revisions in migrations/versions"
index = package.joinpath("server", "static", "index.html")
if expect_dashboard:
    assert index.is_file(), "server/static/index.html missing: build the dashboard (make ui)"
print(f"agent_replay {agent_replay.__version__} from {location}")
print(f"migrations: {', '.join(revisions)}; dashboard bundled: {index.is_file()}")
PY

step "the run store migrates to head"
"$venv/replay" db upgrade --db "sqlite:///$tmp/store.db"

step "replay show and replay run on the toy cassette"
# The toy agent is example code, not part of the package: copy just it next to the
# cassette so `replay run` can import it from the working directory.
mkdir -p "$tmp/work/examples/toy_agent"
cp "$repo/examples/toy_agent/cassettes/refund_eligible.jsonl" "$tmp/work/"
touch "$tmp/work/examples/__init__.py"
cp "$repo"/examples/toy_agent/{__init__,agent,tools}.py "$tmp/work/examples/toy_agent/"
(
    cd "$tmp/work"
    "$venv/replay" show refund_eligible.jsonl > show.txt
    grep -q "examples.toy_agent.agent:run" show.txt || fail "replay show did not print the agent"
    echo "replay show: $(wc -l < show.txt) lines"
    REPLAY_DB_URL="sqlite:///$tmp/store.db" "$venv/replay" run refund_eligible.jsonl
)

step "replay serve"
port=$("$venv/python" -I -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$venv/replay" serve --db "sqlite:///$tmp/serve.db" --port "$port" > "$tmp/serve.log" 2>&1 &
server=$!
base="http://127.0.0.1:$port"
for _ in $(seq 60); do
    curl -sf "$base/api/health" > /dev/null 2>&1 && break
    kill -0 "$server" 2>/dev/null || { cat "$tmp/serve.log"; fail "replay serve exited"; }
    sleep 0.5
done
health=$(curl -sf "$base/api/health") || { cat "$tmp/serve.log"; fail "no answer from $base/api/health"; }
echo "/api/health: $health"
"$venv/python" -I -c '
import json, sys
health = json.loads(sys.argv[1])
assert health["status"] == "ok" and health["version"] == sys.argv[2], health
assert health["backend"] == "sqlite" and health["revision"], health
' "$health" "$version"
if [ "$expect_dashboard" -eq 1 ]; then
    curl -sf "$base/" | grep -q '<div id="root">' || fail "/ did not serve the dashboard"
    echo "/: the dashboard's index.html"
else
    echo "/: HTTP $(curl -s -o /dev/null -w '%{http_code}' "$base/") (no dashboard expected)"
fi
grep -m 1 "serving the" "$tmp/serve.log" || true
kill "$server"
wait "$server" 2>/dev/null || true
server=""

step "sdist: fresh venv, pip install $(basename "$sdist")"
"$python" -m venv "$tmp/venv-sdist"
"$tmp/venv-sdist/bin/python" -m pip install --quiet --disable-pip-version-check "$sdist"
out=$("$tmp/venv-sdist/bin/replay" --version)
echo "$out"
[ "$out" = "agent-replay $version" ] || fail "the sdist's replay --version printed '$out'"

printf '\ncheck_wheel: OK (%s)\n' "$version"
