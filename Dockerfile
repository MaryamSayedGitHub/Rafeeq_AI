FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
# --extra knowledge: the embedder (bge-m3) and FAISS, without them uploaded PDFs are never indexed
RUN uv sync --locked --no-dev --no-install-project --extra knowledge
COPY rafeeq ./rafeeq
COPY contracts ./contracts
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "rafeeq.main:app", "--host", "0.0.0.0", "--port", "8000"]
