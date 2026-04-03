FROM python:3.10-slim-bullseye

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/
 
RUN --mount=type=cache,target=/root/.cache apt-get update && \
    apt-get install -y -qq \
    ffmpeg \
    espeak \
    libportaudio2 \
    build-essential \
    cmake \
    python3-dev \
    portaudio19-dev \
    python3-pyaudio

COPY ./pyproject.toml ./uv.lock ./
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
COPY . .

VOLUME ["/app/models", "/app/output", "/app/datasets", "/app/logs", "/app/songs", "/app/.cache" ]

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# RVC server
EXPOSE 5555

# Webui server
EXPOSE 8501

# Tensorboard server
EXPOSE 6006

ENV PATH="/app/.venv/bin:$PATH:/app/:/"

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
