#!/bin/sh
# Entry point for Dockerfile.allinone: start the API in the background, wait until it is ready,
# then run Streamlit in the foreground on $PORT (7860 on Hugging Face, set by the platform on Render).
set -e

gunicorn --chdir /app/api --workers 1 --threads 4 --bind 127.0.0.1:5000 --timeout 60 app:app &

echo "Waiting for API..."
python - <<'EOF'
import time, urllib.request
for _ in range(180):   # tiny free-tier CPUs (Render: 0.1 CPU) need a while to import + load models
    try:
        urllib.request.urlopen("http://127.0.0.1:5000/ready", timeout=2)
        print("API ready")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("API did not become ready in 180 s")
EOF

# enableXsrfProtection/CORS off: needed for file uploads behind the Hugging Face Spaces proxy.
exec streamlit run /app/ui/streamlit_app.py \
    --server.port="${PORT:-7860}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableXsrfProtection=false \
    --server.enableCORS=false \
    --browser.gatherUsageStats=false
