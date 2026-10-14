# One image for the dashboard: the React build served by `replay serve`, with the
# package, its server and PostgreSQL extras, and the retail example for demo data.

FROM node:22-slim AS ui
WORKDIR /src/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
# vite writes the build to ../agent_replay/server/static
RUN npm run build

FROM python:3.12-slim AS app
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --extra server --extra postgres --no-install-project
COPY agent_replay/ agent_replay/
RUN uv sync --locked --no-dev --extra server --extra postgres
COPY --from=ui /src/agent_replay/server/static/ agent_replay/server/static/
COPY examples/ examples/
COPY eval/results/judge_smoke.json eval/results/judge_smoke.json
RUN useradd --create-home --uid 10001 replay && chown -R replay /app
USER replay
ENV REPLAY_LOG_FORMAT=json
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"]
CMD ["replay", "serve", "--host", "0.0.0.0", "--port", "8000"]
