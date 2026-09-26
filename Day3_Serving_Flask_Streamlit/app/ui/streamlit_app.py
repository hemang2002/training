"""
streamlit_app.py - web UI for the CIFAR-10 classifier API.

The UI does NOT load the model. It is a pure client of the Flask API over HTTP.
That separation is what lets us scale/deploy the API and the UI independently on Day 4.

Run:  streamlit run streamlit_app.py
Config: API_URL env var (default http://127.0.0.1:5000)
        - docker compose : http://api:5000        (service name, not localhost!)
        - kubernetes     : http://cifar-api:5000  (Service name)
"""
from __future__ import annotations

import io
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")  # 127.0.0.1, not localhost (IPv6 delay on Windows)
SAMPLES_DIR = Path(__file__).parent / "samples"

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🧠", layout="wide")


# ----------------------------------------------------------------------------- API helpers
def api_get(base: str, path: str, timeout: float = 5.0):
    r = requests.get(f"{base}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post_image(base: str, path: str, image_bytes: bytes, params: dict | None = None, timeout: float = 30.0):
    files = {"file": ("image.png", image_bytes, "image/png")}
    r = requests.post(f"{base}{path}", files=files, params=params or {}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.json().get('error', r.text)}")
    return r.json()


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
api_url = st.sidebar.text_input("API URL", DEFAULT_API_URL).rstrip("/")

try:
    ready = api_get(api_url, "/ready")
    info = api_get(api_url, "/models")
    st.sidebar.success(f"API ready · models: {', '.join(ready['models'])}")
except Exception as exc:  # noqa: BLE001 - show any connection problem to the user
    st.sidebar.error(f"API not reachable at {api_url}\n\n{exc}")
    st.title("🧠 CIFAR-10 Image Classifier")
    st.warning("Start the API first:  `waitress-serve --port=5000 app:app` (Windows) or "
               "`gunicorn -b 0.0.0.0:5000 app:app` (Linux/macOS) inside `app/api`.")
    st.stop()

variants = list(info["variants"].keys())
default_idx = next((i for i, v in enumerate(variants) if info["variants"][v].get("default")), 0)
model = st.sidebar.selectbox("Model variant", variants, index=default_idx)
top_k = st.sidebar.slider("Top-k", 1, len(info["labels"]), 3)
st.sidebar.caption("Classes: " + ", ".join(info["labels"]))

# ----------------------------------------------------------------------------- input image
st.title("🧠 CIFAR-10 Image Classifier")
st.caption("MobileNetV2 fine-tuned on CIFAR-10 · served by Flask + ONNX Runtime · UI by Streamlit")

col_in, col_out = st.columns([1, 2])
with col_in:
    st.subheader("1. Choose an image")
    source = st.radio("Source", ["Sample image", "Upload"], horizontal=True)
    image = None
    if source == "Upload":
        up = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if up is not None:
            image = Image.open(up)
    else:
        samples = sorted(SAMPLES_DIR.glob("*.png"))
        if samples:
            choice = st.selectbox("Sample", samples, format_func=lambda p: p.stem)
            image = Image.open(choice)
        else:
            st.info("No sample images found - upload one instead.")
    if image is not None:
        # CIFAR images are 32x32; upscale with NEAREST only for display so pixels stay visible.
        st.image(image.convert("RGB").resize((256, 256), Image.NEAREST), caption=f"original size {image.size}")

if image is None:
    st.stop()

image_bytes = to_png_bytes(image)

tab_pred, tab_cmp, tab_bench, tab_card = st.tabs(
    ["🔮 Predict", "⚖️ Compare fp32 vs int8", "⏱️ Benchmark", "📇 Model card"])

# ----------------------------------------------------------------------------- predict
with tab_pred:
    try:
        res = api_post_image(api_url, "/predict", image_bytes, {"model": model, "top_k": top_k})
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", res["prediction"])
        c2.metric("Confidence", f"{res['confidence'] * 100:.1f}%")
        c3.metric("Inference time", f"{res['inference_ms']:.1f} ms")
        probs = pd.DataFrame({"probability": res["probabilities"]}).sort_values("probability", ascending=False)
        st.bar_chart(probs)
        st.dataframe(pd.DataFrame(res["top_k"]))
        st.caption(f"Served by pod: `{res.get('pod')}` · total server time {res['total_ms']} ms")
        with st.expander("Raw JSON response"):
            st.json(res)

# ----------------------------------------------------------------------------- compare
with tab_cmp:
    st.write("The same image is sent through **every** model variant. "
             "Does quantization change the answer? How much faster is it?")
    try:
        cmp = api_post_image(api_url, "/compare", image_bytes)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Compare failed: {exc}")
    else:
        rows = []
        for name, r in cmp["results"].items():
            rows.append({"variant": name, "prediction": r["prediction"],
                         "confidence": r["confidence"], "inference_ms": r["inference_ms"],
                         "size_mb": info["variants"][name]["size_mb"],
                         "test_accuracy": info["variants"][name].get("accuracy")})
        st.dataframe(pd.DataFrame(rows).set_index("variant"))
        if cmp["agree"]:
            st.success("✅ All variants agree on the top-1 class.")
        else:
            st.warning("⚠️ Variants disagree - quantization changed the decision for this image.")
        probs = pd.DataFrame({name: r["probabilities"] for name, r in cmp["results"].items()})
        st.bar_chart(probs)

# ----------------------------------------------------------------------------- benchmark
with tab_bench:
    st.write("Send many requests and look at the latency distribution **as seen by the client** "
             "(network + (de)serialization + preprocessing + inference).")
    c1, c2 = st.columns(2)
    n_req = c1.number_input("Requests", 10, 1000, 50, step=10)
    conc = c2.number_input("Concurrency", 1, 32, 4)
    if st.button("Run benchmark", type="primary"):
        def one(variant: str) -> float:
            t0 = time.perf_counter()
            api_post_image(api_url, "/predict", image_bytes, {"model": variant})
            return (time.perf_counter() - t0) * 1000

        summary = []
        progress = st.progress(0.0)
        for i, v in enumerate(variants):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=int(conc)) as pool:
                lat = sorted(pool.map(one, [v] * int(n_req)))
            wall = time.perf_counter() - t0
            summary.append({"variant": v,
                            "p50_ms": round(statistics.median(lat), 1),
                            "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
                            "max_ms": round(lat[-1], 1),
                            "throughput_rps": round(len(lat) / wall, 1)})
            progress.progress((i + 1) / len(variants))
        df = pd.DataFrame(summary).set_index("variant")
        st.dataframe(df)
        st.bar_chart(df[["p50_ms", "p95_ms"]])
        st.caption("Discuss: why is p95 higher than p50? What changes when concurrency > CPU cores?")

# ----------------------------------------------------------------------------- model card
with tab_card:
    st.json(info)
    try:
        st.subheader("Live server metrics")
        st.json(api_get(api_url, "/metrics"))
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
