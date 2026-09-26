# ---------------------------------------------------------------------------
# Streamlit UI image. Build from the ml-app/ folder:
#     docker build -f ui/Dockerfile -t cifar-ui:v1 .
# The UI never loads the model - it only needs API_URL to reach the API.
# ---------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_URL=http://api:5000

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY ui/requirements.txt .
RUN pip install -r requirements.txt
COPY ui/streamlit_app.py ./
COPY ui/samples/ ./samples/
# Numeric UID (not the name) so Kubernetes runAsNonRoot can verify the user is non-root.
USER 1000

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
