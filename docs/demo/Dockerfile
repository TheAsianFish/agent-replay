# VHS plus uv, so docs/demo.tape can run `uv run replay ...` inside the container.
FROM ghcr.io/charmbracelet/vhs
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /usr/local/bin/
