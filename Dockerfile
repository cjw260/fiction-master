FROM node:24-bookworm-slim AS frontend
RUN corepack enable
WORKDIR /app
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY front/package.json front/pnpm-lock.yaml ./front/
RUN pnpm install --frozen-lockfile
COPY front ./front
RUN pnpm --dir front build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_NO_DEV=1
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./backend/
RUN cd backend && uv sync --frozen --no-dev
COPY backend ./backend
COPY --from=frontend /app/front/dist ./front/dist
RUN mkdir -p /app/fiction /app/data
EXPOSE 8000
CMD ["backend/.venv/bin/uvicorn", "fiction_master.main:app", "--app-dir", "backend/src", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
