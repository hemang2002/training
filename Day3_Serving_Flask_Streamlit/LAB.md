# Day 3 — Server-side Lab (Flask + Streamlit)

Work in pairs. Copy `app/` to `my_app/` first so the reference code stays intact:

```bash
cp -r app my_app          # Windows PowerShell: Copy-Item -Recurse app my_app
```

After each exercise, run the tests: `pytest -q tests` (they import from `app/api`, so point them at your
copy by editing `API_DIR` at the top of `tests/test_api.py`, or just test manually with curl / the notebook).

| # | Exercise | Level | Time |
|---|----------|-------|------|
| 1 | Batch prediction endpoint | ★ | 20 min |
| 2 | API-key authentication | ★ | 15 min |
| 3 | Structured request logging | ★★ | 15 min |
| 4 | Prediction history page in Streamlit | ★★ | 20 min |
| 5 | Warm-up at start-up | ★★ | 10 min |
| 6 | (stretch) Add a new model variant without touching code | ★★★ | 15 min |

---

## Exercise 1 — `POST /predict/batch`

Accept several files in one request (`files=[("files", f1), ("files", f2), ...]`) and run **one**
ONNX call with a batch of N images (our ONNX model has a dynamic batch axis).
Return a list of predictions. Compare the time of 16 single requests vs 1 batch request of 16.

<details><summary>✅ Solution</summary>

In `model_service.py`:

```python
def predict_batch(self, images: list[bytes], model: str | None = None) -> dict:
    model = model or self.default_model
    x = np.concatenate([self.preprocess(b) for b in images], axis=0)   # (N,3,96,96)
    sess = self.sessions[model]
    t0 = time.perf_counter()
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    ms = (time.perf_counter() - t0) * 1000
    probs = self.softmax(logits)
    return {"model": model, "batch_size": len(images), "inference_ms": round(ms, 2),
            "predictions": [{"prediction": self.labels[int(p.argmax())],
                             "confidence": round(float(p.max()), 4)} for p in probs]}
```

In `app.py` (inside `create_app`):

```python
@app.post("/predict/batch")
def predict_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "send one or more files in field 'files'"}), 400
    if len(files) > 64:
        return jsonify({"error": "max 64 images per batch"}), 400
    model = request.args.get("model") or svc.default_model
    if model not in svc.sessions:
        return jsonify({"error": f"unknown model '{model}'"}), 404
    try:
        return jsonify(svc.predict_batch([f.read() for f in files], model))
    except InvalidImageError as exc:
        return jsonify({"error": str(exc)}), 400
```

Client:

```python
files = [("files", (p.name, p.read_bytes(), "image/png")) for p in samples[:16]]
requests.post("http://localhost:5000/predict/batch", files=files).json()
```

**Discussion:** batching raises throughput (better CPU vectorization, fewer HTTP round-trips) but
increases latency for the first image in the batch. This is the core trade-off behind "dynamic batching"
in Triton / TorchServe.
</details>

---

## Exercise 2 — API-key authentication

Require header `X-API-Key` on `/predict` and `/compare`. The expected key comes from env var `API_KEY`.
If `API_KEY` is not set, auth is disabled (so the rest of the course still works). `/health` and `/ready`
must stay public — why?

<details><summary>✅ Solution</summary>

```python
import hmac
from functools import wraps

def require_api_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expected = os.getenv("API_KEY")
        if expected:
            given = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(given, expected):      # constant-time compare
                return jsonify({"error": "invalid or missing API key"}), 401
        return fn(*args, **kwargs)
    return wrapper

@app.post("/predict")
@require_api_key
def predict(): ...
```

Health/readiness stay public because Kubernetes probes and load balancers call them without credentials.
In Streamlit, send the header: `requests.post(url, files=..., headers={"X-API-Key": os.getenv("API_KEY", "")})`.
On Day 4 the key goes into a Kubernetes **Secret**, never into the image.
</details>

---

## Exercise 3 — Structured request logging

Log one JSON line per request: timestamp, path, status, model, latency_ms, client IP.
Use Flask's `before_request` / `after_request` hooks.

<details><summary>✅ Solution</summary>

```python
import json as _json
from flask import g

@app.before_request
def start_timer():
    g.t0 = time.perf_counter()

@app.after_request
def log_request(response):
    if request.path not in ("/health", "/ready"):          # don't spam logs with probe calls
        log.info(_json.dumps({
            "path": request.path, "method": request.method, "status": response.status_code,
            "model": request.args.get("model"), "latency_ms": round((time.perf_counter() - g.t0) * 1000, 2),
            "ip": request.headers.get("X-Forwarded-For", request.remote_addr),
        }))
    return response
```

**Why JSON logs?** Log aggregators (Loki, CloudWatch, Stackdriver, ELK) can filter/aggregate fields
(`status>=500`, `avg(latency_ms) by model`) without regex.
</details>

---

## Exercise 4 — Prediction history in Streamlit

Keep the last 10 predictions (thumbnail, model, label, confidence) in `st.session_state` and display
them in a new tab "🕘 History".

<details><summary>✅ Solution</summary>

```python
if "history" not in st.session_state:
    st.session_state.history = []

# after a successful prediction in tab_pred:
st.session_state.history.insert(0, {"model": res["model"], "prediction": res["prediction"],
                                    "confidence": res["confidence"],
                                    "thumb": image.convert("RGB").resize((64, 64))})
st.session_state.history = st.session_state.history[:10]

# new tab
with tab_hist:
    for h in st.session_state.history:
        c1, c2 = st.columns([1, 4])
        c1.image(h["thumb"])
        c2.write(f"**{h['prediction']}** ({h['confidence']:.0%}) · model `{h['model']}`")
```

Note: Streamlit re-runs the *whole script* on every interaction; `session_state` is what survives re-runs.
Because our script predicts on every re-run, dedupe by checking the image hash before inserting.
</details>

---

## Exercise 5 — Warm-up at start-up

The first inference on a fresh ONNX session is slower (memory allocation, kernel selection).
Run one dummy inference per variant when the service starts, and log the cold vs warm time.

<details><summary>✅ Solution</summary>

At the end of `ModelService.__init__`:

```python
dummy = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
for name, sess in self.sessions.items():
    t0 = time.perf_counter(); sess.run(None, {sess.get_inputs()[0].name: dummy}); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); sess.run(None, {sess.get_inputs()[0].name: dummy}); warm = time.perf_counter() - t0
    log.info("%s warm-up: cold %.1f ms, warm %.1f ms", name, cold * 1000, warm * 1000)
```

Link to Day 4: this is exactly why the Kubernetes **readinessProbe** should only pass *after* warm-up.
</details>

---

## Exercise 6 (stretch) — New variant via config only

Our service reads the list of variants from `models/model_card.json`. Add a third variant
(for example the Day-2 lab's ResNet18 ONNX export, or an fp16 ONNX model) **without changing Python code**:
drop the file into `models/`, add an entry under `"variants"`, restart the API, and check that
`/models`, `/compare` and the Streamlit dropdown all pick it up.

<details><summary>✅ Solution idea</summary>

Make an fp16 variant (`pip install onnxconverter-common` first; it prints harmless
"float32 number … will be truncated" warnings for tiny weights):

```python
import onnx
from onnxconverter_common import float16
m = onnx.load("models/model_fp32.onnx")
m16 = float16.convert_float_to_float16(m, keep_io_types=True)   # inputs/outputs stay float32
onnx.save(m16, "models/model_fp16.onnx")
```

Add to `model_card.json`:

```json
"fp16": {"file": "model_fp16.onnx"}
```

Restart → the new variant appears everywhere (verified: fp16 file 4.5 MB vs 8.9 MB, `cat_1` confidence 0.8695
vs 0.8686 for fp32). **Discussion:** fp16 halves the size, but on most CPUs it is
*not faster* (no native fp16 math) — on GPUs it usually is. "Smaller" ≠ "faster".
</details>
