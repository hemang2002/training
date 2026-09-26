# Day 3 — From Notebook to Service: Flask API + Streamlit UI

> **Theme of the day:** a model in a `.ipynb` helps nobody. Today it becomes a **service** with a
> contract (endpoints, formats, status codes) plus a **UI** that people can actually use.

## 🎯 Learning objectives

By the end of the day students can:

1. Explain the client–server model, REST, HTTP methods and status codes.
2. Wrap a trained model in a Flask API with health/readiness endpoints, input validation and errors.
3. Explain why we serve **ONNX Runtime** instead of PyTorch (dependency size, speed, portability).
4. Keep training and serving preprocessing identical, and show what breaks when they differ.
5. Build a Streamlit front-end that talks to the API over HTTP.
6. Measure latency correctly (inference vs server vs client; p50 vs p95) and compare fp32 vs int8 **inside the service**.
7. Run the API with a production WSGI server (gunicorn / waitress) instead of `app.run()`.

## 📁 What's in this folder

```
Day3_Serving_Flask_Streamlit/
├── README.md                     ← this guide
├── LAB.md                        ← server-side exercises + solutions
├── 01_consuming_the_api.ipynb    ← lecture notebook: call every endpoint, errors, benchmarks, pitfalls
├── 02_student_lab_client.ipynb   ← client-side exercises + solutions
├── app/
│   ├── api/
│   │   ├── app.py                ← Flask routes (/health /ready /models /metrics /predict /compare)
│   │   ├── model_service.py      ← ONNX Runtime loading, preprocessing, softmax, top-k
│   │   └── requirements.txt      ← flask, onnxruntime, numpy, pillow, gunicorn|waitress  (NO torch)
│   └── ui/
│       ├── streamlit_app.py      ← UI: predict, compare fp32 vs int8, benchmark, model card
│       ├── make_samples.py       ← exports CIFAR-10 test images to ui/samples/
│       ├── samples/              ← 20 sample images (2 per class)
│       └── requirements.txt
└── tests/test_api.py             ← pytest: runs with a fake ONNX model, plus real-model checks
```

The API reads the models from `<repo>/models/` (produced on Day 2). If they are missing, run:

```bash
python Day2_Transfer_Learning_and_Optimization/scripts/train_and_export.py
```

## 🕐 Timetable (5 hours)

| Time | Block | Type | Material |
|------|-------|------|----------|
| 0:00 – 0:15 | Recap of Day 2 (what's in `models/`, fp32 vs int8 results) | Discussion | `models/model_card.json` |
| 0:15 – 0:50 | **Serving concepts**: client/server, REST, HTTP, JSON, status codes, why ONNX Runtime | Lecture | §1–§3 below |
| 0:50 – 1:30 | **Code walk-through**: `model_service.py`, `app.py`; start API; curl every endpoint | Live coding | `app/api` |
| 1:30 – 1:45 | ☕ Break | | |
| 1:45 – 2:25 | **Notebook 1**: consume the API, errors, fp32 vs int8 in the service, load test, preprocessing bug, OOD | Live coding | `01_consuming_the_api.ipynb` |
| 2:25 – 3:05 | 🧑‍💻 **Student lab A** — client side | Hands-on | `02_student_lab_client.ipynb` |
| 3:05 – 3:20 | ☕ Break | | |
| 3:20 – 3:50 | **Streamlit**: execution model, widgets, `session_state`, calling the API; run the UI | Lecture + live | `app/ui/streamlit_app.py` |
| 3:50 – 4:40 | 🧑‍💻 **Student lab B** — server side (pick 2–3 exercises) | Hands-on | `LAB.md` |
| 4:40 – 5:00 | Tests (`pytest`), dev server vs gunicorn/waitress, recap, preview of Day 4 | Wrap-up | `tests/` |

Hands-on time: ~1 h 30 min of 5 h (plus students follow along during live coding).
**If running late:** skip Streamlit benchmark tab and LAB exercise 6. **If early:** do LAB exercise 1 (batching) together and discuss dynamic batching.

---

## ▶️ How to run (trainer checklist — do this before class)

```bash
# from repo root, with the course venv active
pip install -r Day3_Serving_Flask_Streamlit/app/api/requirements.txt -r Day3_Serving_Flask_Streamlit/app/ui/requirements.txt

# 1) API  (terminal 1)
cd Day3_Serving_Flask_Streamlit/app/api
waitress-serve --port=5000 app:app            # Windows
gunicorn -w 2 -b 0.0.0.0:5000 app:app         # Linux / macOS
# dev only: python app.py

# 2) Smoke test (terminal 2)
curl http://localhost:5000/health
curl http://localhost:5000/models
curl -F "file=@../ui/samples/cat_1.png" "http://localhost:5000/predict?model=int8"
curl -F "file=@../ui/samples/cat_1.png" http://localhost:5000/compare

# 3) UI  (terminal 2)
cd ../ui
streamlit run streamlit_app.py                # opens http://localhost:8501

# 4) Tests (from Day3 folder)
pytest -q tests
```

> **Windows note:** in PowerShell, `curl` is an alias for `Invoke-WebRequest`. Use `curl.exe` instead.

---

## 1. Serving concepts (lecture notes)

### 1.1 Why a service?

A notebook is single-user, manual, and tied to one machine. A service is:

* **Callable** by any program in any language (mobile app, website, another service).
* **Versioned** — clients depend on a contract, not on your code.
* **Scalable** — more copies behind a load balancer (Day 4).
* **Observable** — logs, metrics, health checks.

### 1.2 REST in 5 minutes

| Concept | Our API |
|---|---|
| Resource | `/models`, `/predict` |
| Method | `GET` = read, no side effects · `POST` = send data to be processed |
| Request body | `multipart/form-data` (file upload) or `application/json` (base64) |
| Response | JSON |
| Status codes | `200` OK · `400` bad input · `401` unauthorized · `404` not found · `413` too large · `500` server bug · `503` not ready |

Talking point: status codes are for **machines** — a client can retry on `503` but must not retry on `400`.

### 1.3 Why ONNX Runtime, not PyTorch, in the API?

| | PyTorch in API | ONNX Runtime in API |
|---|---|---|
| pip install size | ~200 MB (CPU) · 2 GB+ (CUDA) | ~15–20 MB |
| Docker image | ~1–2.5 GB | ~250–350 MB |
| Needs model **code** (class definition) | yes | no — graph is self-contained |
| Graph optimizations (fusion, constant folding) | limited in eager | built in |
| Runs INT8 QDQ models | via torch.ao (CPU backends) | yes, CPU + other providers |
| Language support | Python, C++ | Python, C++, C#, Java, JS, … |

The model is **trained** in PyTorch and **served** with ONNX Runtime. Same weights, different runtime.

### 1.4 Anatomy of a request (show this on the whiteboard)

```
client ──HTTP──▶ WSGI server (gunicorn/waitress)
                   └─▶ Flask route /predict
                         ├─ read bytes (multipart / base64)          ← validation → 400
                         ├─ decode image (Pillow) + RGB convert      ← invalid → 400
                         ├─ resize 96×96, /255, normalize, NCHW      ← MUST match training
                         ├─ session.run()  (ONNX Runtime)            ← "inference_ms"
                         ├─ softmax + top-k
                         └─ JSON response                             ← "total_ms"
client measures round-trip                                            ← "client ms"
```

### 1.5 Code walk-through: `model_service.py`

Points to highlight while scrolling through the file:

1. **`find_model_dir()`** — same code works in three places (Day 3 folder, Day 4 folder, Docker) thanks to
   the `MODEL_DIR` env var + a fallback search. *Configuration via environment variables* is a
   12-factor app principle and pays off on Day 4.
2. **`SessionOptions`** — `ORT_ENABLE_ALL` turns on graph optimizations; `intra_op_num_threads` from
   `ORT_THREADS`. In containers we keep threads low and scale with replicas instead.
3. **`preprocess()`** — `convert("RGB")` handles PNG-with-alpha / grayscale; bilinear resize matches
   torchvision's `Resize` on PIL images; normalization uses mean/std from `model_card.json`
   → one source of truth.
4. **`softmax()`** — subtract the max before `exp` for numerical stability (ask: what happens with logits of 1000?).
5. **Variants come from `model_card.json`** — adding a model is a config change, not a code change.

### 1.6 Code walk-through: `app.py`

1. **App factory `create_app(service)`** — lets tests inject a fake model (see `tests/test_api.py`).
2. **`/health` vs `/ready`** — liveness vs readiness (Day 4 probes). A slow model load must not trigger restarts.
3. **Input validation** — every bad input returns a clear JSON error and the right status code.
4. **`MAX_CONTENT_LENGTH`** — protects memory from giant uploads (413).
5. **`HOSTNAME` in responses** — in Kubernetes this is the pod name, so students can *see* load balancing.
6. **`app.run()` only for development**; production uses gunicorn (Linux) or waitress (Windows).
   Also `host="0.0.0.0"`: inside a container `127.0.0.1` means "only this container" → classic Day 4 bug.

### 1.7 Dev server vs production WSGI server

| | `app.run()` (Werkzeug) | gunicorn / waitress |
|---|---|---|
| Purpose | development, auto-reload, debugger | production |
| Concurrency | threads in 1 process | multiple **worker processes** (+ threads) |
| Robustness | not hardened | restarts dead workers, timeouts |
| Command | `python app.py` | `gunicorn -w 2 -b 0.0.0.0:5000 app:app` |

Each gunicorn worker is a separate process and **loads its own copy of the model** → memory = workers × model size.
Our int8 model is small, so that's fine; for a 7 GB LLM it is not. Discuss.

---

## 2. Streamlit (lecture notes)

### 2.1 Mental model

Streamlit **re-runs the whole script top-to-bottom** on every interaction. Widgets return values.
State that must survive re-runs goes in `st.session_state`; expensive calls go in `@st.cache_data` /
`@st.cache_resource`.

### 2.2 Why the UI calls the API instead of loading the model

```
       ┌─────────────┐    HTTP    ┌─────────────┐
user ─▶│  Streamlit  │──────────▶│  Flask API  │──▶ ONNX model
       └─────────────┘            └─────────────┘
         scales with users          scales with inference load
```

* Separation of concerns: UI people and ML people deploy independently.
* The same API serves the UI, a mobile app, and batch jobs.
* On Day 4 they become **two containers** / **two Deployments**.

### 2.3 Walk-through of `streamlit_app.py`

* Sidebar reads `API_URL` from env (`localhost` locally, `api` in docker-compose, `cifar-api` in k8s).
* If the API is down → friendly error + `st.stop()`.
* Tabs: **Predict**, **Compare fp32 vs int8** (calls `/compare`), **Benchmark** (thread pool load test),
  **Model card** (+ live `/metrics`).

---

## 3. "What difference does optimization make?" — now measured in the service

Run the Compare and Benchmark tabs (or Notebook 1 §6–§7) and fill in with the class:

| Metric | fp32 | int8 | Observation |
|---|---|---|---|
| File size (MB) | | | ~4× smaller weights |
| `inference_ms` (server) | | | faster, but by less than 4× |
| Client p50 / p95 (ms) | | | network + HTTP overhead dilute the gain |
| Accuracy on test set (model card) | | | small drop |
| Agreement on samples | | | |

Key message: **optimizing the model is only one part of optimizing the service.** At small model sizes,
HTTP, JSON, image decoding and Python overhead can dominate.

---

## ⚠️ Common student mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `ConnectionRefusedError` from Streamlit | API not running, or wrong `API_URL` | start API; check sidebar URL |
| `curl` in PowerShell behaves strangely | alias for `Invoke-WebRequest` | use `curl.exe` |
| `gunicorn` fails on Windows | gunicorn is Unix-only (needs `fcntl`) | use `waitress-serve` |
| `/ready` returns 503 | `models/` empty | run Day 2 `scripts/train_and_export.py` |
| Predictions are all wrong but no error | preprocessing mismatch (BGR, wrong mean/std, wrong size) | keep preprocessing in the API; see Notebook 1 §8 |
| File upload returns 413 | image bigger than `MAX_UPLOAD_MB` | resize client-side or raise the limit |
| Changes in `app.py` not visible | server not restarted (no auto-reload with waitress/gunicorn) | restart; or `flask --app app run --debug` during development |
| Port 5000 in use (macOS) | AirPlay Receiver uses 5000 | `--port 5001` and update `API_URL` |

## 💬 Discussion questions (sprinkle through the day)

1. Why is `/predict` a `POST` and not a `GET`?
2. Our API returns probabilities for all 10 classes. When might you *not* want to expose them? (model extraction attacks, privacy)
3. The int8 model is 4× smaller. Why is the end-to-end request not 4× faster?
4. Two gunicorn workers each load the model. What's the memory cost? When does that become a problem?
5. How would you roll out a new model version without breaking clients? (versioned endpoints `/v2/predict`, model field in response, canary)
6. What should happen if the model file is corrupted at start-up — crash, or start and report "not ready"?
7. The model says "cat, 97%" on a photo of a car's dashboard. Is the API wrong? Is the model wrong?

## 📝 Recap & homework

**Recap:** contract → validation → identical preprocessing → measure at 3 levels → separate UI from API → production WSGI server.

**Homework:**
1. Finish two LAB exercises you didn't do in class.
2. Read Flask's "Deploying to Production" page and Streamlit's "Main concepts" (links in `References/reading_list.md`).
3. Install Docker Desktop (or Docker Engine) and run `docker run hello-world` — **required for Day 4**.
4. Optional: install `minikube` or `kind`, and run `kubectl version --client`.

**Teaser for Day 4:** *"It works on my machine"* — tomorrow we ship the machine: Docker images for the API and
UI, docker-compose, Kubernetes with two replicas and autoscaling, and a free cloud deployment you can open on your phone.
