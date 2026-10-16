# Terminal demo for the README. Offline: no API key, no network at record time.
#
# Render from the repo root, either with a local install (needs vhs, ttyd,
# ffmpeg and uv):
#
#   vhs docs/demo.tape
#
# or with Docker (nothing to install; the host .venv is left alone):
#
#   docker build -t replay-demo docs/demo
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/vhs" -w /vhs \
#     -e HOME=/tmp/home -e UV_CACHE_DIR=/tmp/uv-cache \
#     -e UV_PROJECT_ENVIRONMENT=/tmp/venv -e UV_PYTHON_INSTALL_DIR=/tmp/uv-python \
#     replay-demo docs/demo.tape
#
# Output: docs/demo.gif

Output docs/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 16
Set Theme "nord"
Set TypingSpeed 35ms
Set Framerate 12

# Setup, not shown: warm the environment so the demo is not spent installing.
Hide
Type "export PS1='$ ' && uv run replay --help > /dev/null && clear"
Enter
Sleep 8s
Show

Type "# Replay a recorded agent run against its checks"
Enter
Type "uv run replay ci examples/retail_agent/replay.toml --no-store"
Sleep 400ms
Enter
Sleep 6s

Hide
Type "clear"
Enter
Show

Type "# Same suite, with a regression seeded into the agent"
Enter
Type "uv run replay ci examples/retail_agent/replay.toml --no-store --agent examples.retail_agent.mutants:wrong_write_argument"
Sleep 400ms
Enter
Sleep 7s

Type "echo exit $?"
Sleep 300ms
Enter
Sleep 4s
