"""
model_service.py - loads the ONNX models produced on Day 2 and runs inference.

Why ONNX Runtime and not PyTorch here?
  * The API only needs *inference*, not training.
  * `onnxruntime` + `numpy` + `pillow` is ~60 MB; `torch` is ~200 MB (CPU) to 2+ GB (CUDA).
  * Smaller dependency -> smaller Docker image -> faster deploys, cheaper hosting.

The preprocessing below MUST match training exactly (Day 2 contract):
  RGB -> Resize((96, 96)) bilinear -> scale to [0, 1] -> Normalize(ImageNet mean/std) -> NCHW float32
A mismatch here is the #1 cause of "great in the notebook, bad in production".
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]


def find_model_dir() -> Path:
    """MODEL_DIR env var wins; otherwise walk up from this file looking for models/model_card.json.

    This lets the same code run from Day3/app/api (models live in <repo>/models),
    from Day4/ml-app/api (models live in ml-app/models) and inside Docker (/app/models).
    """
    env = os.getenv("MODEL_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "models"
        if (candidate / "model_card.json").exists() or (candidate / "model_fp32.onnx").exists():
            return candidate
    return here.parent / "models"


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a decodable image."""


class ModelService:
    def __init__(self, model_dir: str | Path | None = None, threads: int | None = None):
        self.model_dir = Path(model_dir) if model_dir else find_model_dir()
        self.card = self._load_card()
        self.labels: list[str] = self.card.get("labels", CIFAR10_LABELS)
        self.img_size: int = int(self.card.get("img_size", 96))
        self.mean = np.array(self.card.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(self.card.get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 3, 1, 1)

        # Threads per session. In containers keep this small and scale with replicas instead.
        threads = threads if threads is not None else int(os.getenv("ORT_THREADS", "0"))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            opts.intra_op_num_threads = threads

        self.sessions: dict[str, ort.InferenceSession] = {}
        variants = self.card.get("variants") or {
            "fp32": {"file": "model_fp32.onnx"},
            "int8": {"file": "model_int8.onnx"},
        }
        for name, meta in variants.items():
            path = self.model_dir / meta["file"]
            if not path.exists():
                log.warning("Model variant %s missing at %s - skipping", name, path)
                continue
            t0 = time.perf_counter()
            self.sessions[name] = ort.InferenceSession(str(path), sess_options=opts,
                                                       providers=["CPUExecutionProvider"])
            log.info("Loaded %s (%s) in %.0f ms", name, path.name, (time.perf_counter() - t0) * 1000)

        self.variant_meta = {k: v for k, v in variants.items() if k in self.sessions}
        default = os.getenv("DEFAULT_MODEL", "int8")
        self.default_model = default if default in self.sessions else next(iter(self.sessions), None)

    # ------------------------------------------------------------------ helpers
    def _load_card(self) -> dict:
        card_path = self.model_dir / "model_card.json"
        if card_path.exists():
            return json.loads(card_path.read_text(encoding="utf-8"))
        labels_path = self.model_dir / "labels.json"
        card = {}
        if labels_path.exists():
            card["labels"] = json.loads(labels_path.read_text(encoding="utf-8"))
        return card

    @property
    def ready(self) -> bool:
        return bool(self.sessions)

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")  # handles PNG with alpha, grayscale, CMYK...
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("File is not a valid image") from exc
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0          # HWC, [0,1]
        x = x.transpose(2, 0, 1)[np.newaxis, ...]               # -> NCHW
        return ((x - self.mean) / self.std).astype(np.float32)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)  # numerically stable
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---------------------------------------------------------------- inference
    def predict(self, image_bytes: bytes, model: str | None = None, top_k: int = 3) -> dict:
        model = model or self.default_model
        if model not in self.sessions:
            raise KeyError(model)
        x = self.preprocess(image_bytes)
        sess = self.sessions[model]
        t0 = time.perf_counter()
        logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        probs = self.softmax(logits)[0]
        top_k = max(1, min(int(top_k), len(self.labels)))
        idx = np.argsort(probs)[::-1][:top_k]
        return {
            "model": model,
            "prediction": self.labels[idx[0]],
            "confidence": round(float(probs[idx[0]]), 4),
            "top_k": [{"label": self.labels[i], "probability": round(float(probs[i]), 4)} for i in idx],
            "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(self.labels, probs)},
            "inference_ms": round(latency_ms, 2),
        }

    def info(self) -> dict:
        out = {}
        for name, meta in self.variant_meta.items():
            path = self.model_dir / meta["file"]
            out[name] = {
                **meta,
                "size_mb": round(path.stat().st_size / 1e6, 2),
                "default": name == self.default_model,
            }
        return {"img_size": self.img_size, "labels": self.labels, "variants": out,
                "trained_on": self.card.get("trained_on"), "created": self.card.get("created")}
