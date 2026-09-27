# Serving a model — the simple version (show this first)

Three small files, **~140 lines in total** (half of it comments), that turn our trained model into a web app.
Every line has a comment in plain words. Show these on the projector **before** opening the full `app/` folder.

| | `simple/` (this folder) | `app/` (full version) |
|---|---|---|
| Purpose | **show and explain** in class | students **try, test and break** on their own |
| Size | ~140 lines, 3 files | ~520 lines, 3 files + tests |
| Endpoints | `/predict` (+ `/` hello) | `/health` `/ready` `/models` `/metrics` `/predict` `/compare` |
| Models | int8 only | fp32 **and** int8, pick per request |
| Port | **5001** | 5000 |

The two versions use different ports, so both can run at the same time.

## The idea in one picture

```
  you pick a picture         the API runs the model           you see the answer
┌──────────────────┐  HTTP  ┌──────────────────────┐        ┌──────────────────┐
│ Streamlit page   │──────▶│ Flask API            │──────▶ │ "cat, 85 % sure" │
│ step3_simple_ui  │◀──────│ step2_predict_api    │        │  + a bar chart   │
└──────────────────┘  JSON  └──────────────────────┘        └──────────────────┘
```

* **API** = a program that waits for questions and sends back answers.
* **JSON** = the answer written as text that any program can read: `{"label": "cat", "confidence": 0.85}`.
* **Port** = a door number on your computer. Our API waits behind door 5001.

## Run it (from this folder, course venv active)

**Step 1 — the model API** (15 min)
```bash
python predict_api.py
```
In a second terminal:
```bash
curl.exe -F "file=@../app/ui/samples/cat_1.png" http://127.0.0.1:5001/predict
```
Answer: `{"label": "cat", "confidence": 0.85, "all_probabilities": {...}}`

Walk through the file top to bottom — it is exactly 3 blocks:
1. **load the model once** (not on every request — loading is slow),
2. **picture → numbers** (must match training *exactly*),
3. **the `/predict` route** (numbers → model → probabilities → answer).

**Step 2 — the web page** (10 min) — keep step 2 running
```bash
streamlit run simple_ui.py
```
Pick a sample, press **🔮 Ask the model**. Then stop the API and press the button again → friendly error, no crash.

## Things to try live (each takes 1 minute)

| Change | What happens | Lesson |
|---|---|---|
| Delete `/ 255.0` in step 2 | almost every picture becomes "cat, ~52 %" — only 2 of 20 samples right, and **no error** | preprocessing must match training |
| Change `(96, 96)` to `(32, 32)` | `500` server error; the terminal says `Got: 32 Expected: 96` | the model has a fixed input shape |
| Send a request with no file (`curl.exe -X POST http://127.0.0.1:5001/predict`) | `400` + friendly message | APIs must check their input |
| Upload a photo of yourself, or random noise | it still picks one of the 10 classes (random noise → "cat, 65 %") | the model only knows 10 things and never says "I don't know" |

(After each experiment, undo the change and restart the API — Flask does not reload by itself.)

## Measured on the reference laptop

* All 20 sample images give **the same label** as the full `app/` API (int8), probabilities equal up to rounding.
* 18 of 20 samples are correct; `bird_2` → cat and `horse_2` → truck are the model's honest mistakes (good discussion).

## Then: "what is missing for real life?"

Ask the class before opening `app/`. Typical answers and where the full version handles them:

| Missing in the simple version | Where in `app/` |
|---|---|
| "Is the server alive? Is the model loaded?" (Kubernetes asks this on Day 4) | `/health`, `/ready` |
| choose fp32 or int8 | `?model=int8` on `/predict`, `/compare` |
| bad file / huge file / not an image | `InvalidImageError`, `MAX_CONTENT_LENGTH` (413) |
| how fast is it? how many requests? | `inference_ms`, `/metrics` |
| settings without editing code (model folder, port) | environment variables `MODEL_DIR`, `PORT` |
| a real server instead of `app.run()` | `waitress-serve` / `gunicorn` |
| automatic tests | `tests/test_api.py` |
