"""
app.py - Flask REST API that serves the CIFAR-10 MobileNetV2 models (fp32 + int8 ONNX).

Endpoints
  GET  /          -> API description
  GET  /health    -> liveness  : "is the process alive?"          (Kubernetes livenessProbe)
  GET  /ready     -> readiness : "are the models loaded?"         (Kubernetes readinessProbe)
  GET  /models    -> available model variants + metadata
  GET  /metrics   -> simple request counters / latency stats
  POST /predict   -> classify one image     (multipart field "file", or JSON {"image_b64": ...})
                     query params: ?model=fp32|int8  &top_k=3
  POST /compare   -> run the SAME image through every variant (fp32 vs int8 side by side)

Run locally
  Linux/macOS : gunicorn -w 2 -b 0.0.0.0:5000 app:app
  Windows     : waitress-serve --port=5000 app:app
  Dev only    : python app.py
"""
from __future__ import annotations

import base64
import binascii
import logging
import os
import threading
import time

from flask import Flask, jsonify, request

from model_service import InvalidImageError, ModelService

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))


class Metrics:
    """Tiny thread-safe in-memory metrics. (Real systems: Prometheus client.)"""

    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.errors = 0
        self.per_model: dict[str, dict] = {}

    def record(self, model: str, ms: float):
        with self._lock:
            self.requests += 1
            m = self.per_model.setdefault(model, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            m["count"] += 1
            m["total_ms"] += ms
            m["max_ms"] = max(m["max_ms"], ms)

    def error(self):
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "requests": self.requests,
                "errors": self.errors,
                "models": {k: {"count": v["count"],
                               "avg_ms": round(v["total_ms"] / v["count"], 2),
                               "max_ms": round(v["max_ms"], 2)}
                           for k, v in self.per_model.items()},
            }


def _read_image_from_request() -> bytes:
    """Accept either multipart/form-data (field 'file') or JSON {"image_b64": "..."}."""
    if "file" in request.files:
        data = request.files["file"].read()
    elif request.is_json and "image_b64" in (request.get_json(silent=True) or {}):
        b64 = request.get_json()["image_b64"]
        if "," in b64:  # strip "data:image/png;base64," prefix if present
            b64 = b64.split(",", 1)[1]
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            raise InvalidImageError("image_b64 is not valid base64")
    else:
        raise InvalidImageError("Send an image as multipart field 'file' or JSON {'image_b64': ...}")
    if not data:
        raise InvalidImageError("Empty file")
    return data


def create_app(service: ModelService | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)

    svc = service or ModelService()
    metrics = Metrics()
    app.extensions["model_service"] = svc
    log.info("Model dir: %s | variants: %s | default: %s",
             svc.model_dir, list(svc.sessions), svc.default_model)

    @app.get("/")
    def index():
        return jsonify({
            "name": "CIFAR-10 image classifier API",
            "endpoints": ["/health", "/ready", "/models", "/metrics", "POST /predict", "POST /compare"],
            "example": "curl -F file=@cat.png 'http://localhost:5000/predict?model=int8'",
            "pod": os.getenv("HOSTNAME", "local"),  # shows which replica answered (useful in k8s)
        })

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/ready")
    def ready():
        if svc.ready:
            return jsonify({"status": "ready", "models": list(svc.sessions)})
        return jsonify({"status": "not ready", "reason": f"no models found in {svc.model_dir}"}), 503

    @app.get("/models")
    def models():
        return jsonify(svc.info())

    @app.get("/metrics")
    def get_metrics():
        return jsonify(metrics.snapshot())

    @app.post("/predict")
    def predict():
        model = request.args.get("model") or svc.default_model
        top_k = request.args.get("top_k", 3)
        if model not in svc.sessions:
            metrics.error()
            return jsonify({"error": f"unknown model '{model}'", "available": list(svc.sessions)}), 404
        try:
            top_k = int(top_k)
        except ValueError:
            metrics.error()
            return jsonify({"error": "top_k must be an integer"}), 400
        try:
            t0 = time.perf_counter()
            result = svc.predict(_read_image_from_request(), model=model, top_k=top_k)
            result["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        except InvalidImageError as exc:
            metrics.error()
            return jsonify({"error": str(exc)}), 400
        metrics.record(model, result["total_ms"])
        result["pod"] = os.getenv("HOSTNAME", "local")
        return jsonify(result)

    @app.post("/compare")
    def compare():
        try:
            data = _read_image_from_request()
            results = {name: svc.predict(data, model=name, top_k=3) for name in svc.sessions}
        except InvalidImageError as exc:
            metrics.error()
            return jsonify({"error": str(exc)}), 400
        for name, r in results.items():
            metrics.record(name, r["inference_ms"])
        preds = {r["prediction"] for r in results.values()}
        return jsonify({"results": results, "agree": len(preds) == 1})

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": f"file too large (max {MAX_UPLOAD_MB} MB)"}), 413

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return jsonify({"error": "method not allowed"}), 405

    return app


app = create_app()

if __name__ == "__main__":
    # Development server only. Never use app.run() in production - use gunicorn/waitress.
    # host="0.0.0.0" matters inside Docker: 127.0.0.1 would only accept connections from inside the container.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
