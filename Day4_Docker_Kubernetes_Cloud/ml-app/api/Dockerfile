# ---------------------------------------------------------------------------
# CIFAR-10 API image (Flask + gunicorn + ONNX Runtime)
# Build from the ml-app/ folder (the build context must contain api/ AND models/):
#     docker build -f api/Dockerfile -t cifar-api:v1 .
# ---------------------------------------------------------------------------

# ---------- Stage 1: builder - install dependencies into an isolated venv ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Copy ONLY requirements first -> this layer is cached until requirements.txt changes,
# so editing app.py does not trigger a slow re-install.
COPY api/requirements.txt .
RUN pip install -r requirements.txt

# ---------- Stage 2: runtime - just the venv + our code + the models ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=/app/models \
    PORT=5000 \
    DEFAULT_MODEL=int8 \
    WORKERS=2 \
    THREADS=2 \
    ORT_THREADS=1

# Run as a non-root user (security best practice; many platforms require it).
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY api/app.py api/model_service.py ./
COPY models/ ./models/
# Numeric UID (not the name) so Kubernetes runAsNonRoot can verify the user is non-root.
USER 1000

EXPOSE 5000
# Docker-level health check (docker ps shows "healthy"). Kubernetes ignores this and uses its own probes.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health', timeout=2)" || exit 1

# exec -> gunicorn becomes PID 1 and receives SIGTERM directly (graceful shutdown in docker stop / k8s).
CMD ["sh", "-c", "exec gunicorn --workers ${WORKERS} --threads ${THREADS} --bind 0.0.0.0:${PORT} --timeout 60 --access-logfile - app:app"]
