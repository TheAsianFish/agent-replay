# One image for the dashboard: the React build served by `replay serve`, with the
# package, its server and PostgreSQL extras, and the retail example for demo data.
# The package is installed as the wheel is (not editable, the dashboard bundled in
# agent_replay/server/static), into a venv copied into a final stage without uv or
# the package sources.
#   docker build --build-arg VERSION=1.0.0 -t agent-replay .

ARG PYTHON_IMAGE=python:3.12-slim

FROM node:22-slim AS ui
WORKDIR /src/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
# vite writes the build to ../agent_replay/server/static
RUN npm run build

FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --extra server --extra postgres --no-install-project
COPY agent_replay/ agent_replay/
COPY --from=ui /src/agent_replay/server/static/ agent_replay/server/static/
RUN uv sync --locked --no-dev --extra server --extra postgres --no-editable \
    && test -f /opt/venv/lib/python3.12/site-packages/agent_replay/server/static/index.html

FROM ${PYTHON_IMAGE} AS runtime
ARG VERSION=dev
LABEL org.opencontainers.image.title="agent-replay" \
      org.opencontainers.image.description="Record-and-replay regression testing for LLM agents: the run store's dashboard and API" \
      org.opencontainers.image.source="https://github.com/TheAsianFish/agent-replay" \
      org.opencontainers.image.url="https://github.com/TheAsianFish/agent-replay" \
      org.opencontainers.image.documentation="https://github.com/TheAsianFish/agent-replay#readme" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REPLAY_LOG_FORMAT=json
COPY --from=build /opt/venv /opt/venv
RUN useradd --create-home --uid 10001 replay && install -d -o replay /app
WORKDIR /app
# Example code and data for the demo seed (python -m examples.demo_data), not the package.
COPY --chown=replay examples/ examples/
COPY --chown=replay eval/golden/retail/ eval/golden/retail/
USER replay
EXPOSE 8000

# The hosted demo (RELEASE.md, step 9): the demo data baked into a SQLite store at build
# time, so the demo is one container with no database or volume. Listens on $PORT.
#   docker build --target demo -t agent-replay-demo .
FROM runtime AS demo
ENV REPLAY_DB_URL=sqlite:////app/demo.db
RUN python -m examples.demo_data --db "$REPLAY_DB_URL"
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["sh", "-c", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT:-8000}/api/health', timeout=2)\""]
CMD ["sh", "-c", "exec replay serve --host 0.0.0.0 --port \"${PORT:-8000}\""]

# The default image (last stage): the dashboard over $REPLAY_DB_URL, as compose.yaml runs it.
FROM runtime AS app
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"]
CMD ["replay", "serve", "--host", "0.0.0.0", "--port", "8000"]
