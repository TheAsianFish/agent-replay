#!/usr/bin/env bash
# Record the scripted dashboard walkthrough and convert it for the README.
#
# Usage (from the repo root, after `make setup` and `npm --prefix dashboard ci`
# plus `npx playwright install chromium` in dashboard/):
#
#   scripts/walkthrough.sh
#
# Environment:
#   PORT=8000          port for the temporary `replay serve`
#   SKIP_DEMO_DATA=1   reuse an existing .replay/demo.db instead of `make demo-data`
#
# Output: docs/walkthrough/walkthrough.mp4 and docs/walkthrough/walkthrough.gif.
# The intermediate walkthrough.webm is deleted at the end.
set -euo pipefail

cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
OUT=docs/walkthrough

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg is not installed; install it (apt-get install ffmpeg / brew install ffmpeg) and re-run" >&2
  exit 1
fi

if [ ! -d dashboard/node_modules ]; then
  npm --prefix dashboard ci --no-audit --no-fund
fi
npm --prefix dashboard run build

if [ "${SKIP_DEMO_DATA:-0}" != "1" ]; then
  make demo-data
elif [ ! -f .replay/demo.db ]; then
  echo "error: SKIP_DEMO_DATA=1 but .replay/demo.db is missing; run \`make demo-data\` first" >&2
  exit 1
fi

uv run replay serve --db sqlite:///.replay/demo.db --port "$PORT" &
server=$!
trap 'rc=$?; kill $server 2>/dev/null || true; wait $server 2>/dev/null || true; exit $rc' EXIT
timeout 60 bash -c "until curl -sf http://127.0.0.1:$PORT/api/health >/dev/null; do sleep 1; done" \
  || { echo "error: replay serve did not become healthy on port $PORT within 60 s" >&2; exit 1; }

mkdir -p "$OUT"
rm -f "$OUT/walkthrough.webm"
DASHBOARD_URL="http://127.0.0.1:$PORT" npm --prefix dashboard run walkthrough
[ -f "$OUT/walkthrough.webm" ] || { echo "error: the walkthrough test did not write $OUT/walkthrough.webm" >&2; exit 1; }

ffmpeg -y -loglevel error -i "$OUT/walkthrough.webm" \
  -c:v libx264 -pix_fmt yuv420p -crf 28 -preset slow -movflags +faststart -an \
  "$OUT/walkthrough.mp4"

# The GIF is made from the mp4: the webm encoder noise bloats the palette-dithered GIF (~12 MB vs ~3 MB).
palette_dir="$(mktemp -d)"
palette="$palette_dir/palette.png"
filters="fps=10,scale=960:-1:flags=lanczos"
ffmpeg -y -loglevel error -i "$OUT/walkthrough.mp4" -vf "$filters,palettegen=stats_mode=diff:max_colors=96" "$palette"
ffmpeg -y -loglevel error -i "$OUT/walkthrough.mp4" -i "$palette" \
  -lavfi "$filters [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  "$OUT/walkthrough.gif"
rm -rf "$palette_dir"; rm -f "$OUT/walkthrough.webm"

ls -l "$OUT"
