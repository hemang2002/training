# Train → Optimize → Serve → Containerize → Deploy
### A 5-day hands-on course on the full life of a PyTorch model (MSc level, ≤ 5 h/day)

Students take **one model through its whole life**: train it, fine-tune a pretrained one, make it smaller
and faster (and measure what that changes), serve it behind a Flask API with a Streamlit UI, package it in
Docker, run it on Kubernetes, and deploy it to a free cloud tier. The last day trains them to **think** like
ML engineers through questions, debugging cases and design exercises.

```
Day 1  PyTorch fundamentals ──► CNN on FashionMNIST (from scratch)
Day 2  Transfer learning   ──► MobileNetV2 on CIFAR-10 ──► quantize / prune / ONNX ──► models/*.onnx
       + extras            ──► augmentation · segmentation (U-Net) · RNN/LSTM text
Day 3  Serving             ──► Flask API (ONNX Runtime) + Streamlit UI      (uses models/ from Day 2)
Day 4  Shipping            ──► Docker ──► docker-compose ──► Kubernetes ──► free cloud
Day 5  Thinking            ──► question bank, debugging scenarios, system design, Fermi estimates, quiz
Extra  Live demos          ──► object detection (Faster R-CNN) + GAN (DCGAN), trained AND optimized on CPU
```

## 📁 Folder map

| Folder | What's inside | Start with |
|---|---|---|
| [`Day1_PyTorch_Fundamentals/`](Day1_PyTorch_Fundamentals/) | tensors & autograd, training loop, CNN on FashionMNIST, lab | `README.md` |
| [`Day2_Transfer_Learning_and_Optimization/`](Day2_Transfer_Learning_and_Optimization/) | MobileNetV2 fine-tuning, dynamic/static/QAT quantization, pruning, FP16, ONNX + ONNX Runtime INT8, lab, one-command `scripts/train_and_export.py`; extra notebooks: data augmentation, image segmentation (Penn-Fudan, U-Net vs pretrained LR-ASPP), RNN/LSTM text classification (SMS spam) | `README.md` |
| [`Day3_Serving_Flask_Streamlit/`](Day3_Serving_Flask_Streamlit/) | Flask API, Streamlit UI, tests, API notebooks, server-side LAB | `README.md` |
| [`Day4_Docker_Kubernetes_Cloud/`](Day4_Docker_Kubernetes_Cloud/) | Dockerfiles (good vs naive), compose, Kubernetes manifests + load test, free-cloud guides, LAB | `README.md` |
| [`Day5_Thinking_Like_an_ML_Engineer/`](Day5_Thinking_Like_an_ML_Engineer/) | 106 questions with answers, 15 debugging scenarios, 5 design cases, 10 Fermi problems, 30-question quiz | `README.md` |
| [`Live_Demos/`](Live_Demos/) | detection (Penn-Fudan) + optimization, DCGAN + optimization, cached checkpoints as a safety net | `README.md` |
| [`Training_Visualized/`](Training_Visualized/) | 16 interactive, offline HTML lessons (model → gradient descent → backprop → training loop → activations & softmax → NN playground → overfitting → dropout & BatchNorm → metrics → CNN → transfer learning → quantization/pruning/distillation → data augmentation → image segmentation → RNN & LSTM → reinforcement learning), Simple/Detailed view | `index.html` |
| [`Docker_K8s_Visualized/`](Docker_K8s_Visualized/) | 13 interactive, offline HTML pages for Day 4 (install Docker Desktop → why containers → images & layers → running containers → compose → why Kubernetes → Pod → Deployment → Service → ConfigMap/Secret → probes & resources → autoscaling → hands-on lab with real command output) | `index.html` |
| [`CICD_ML_Visualized/`](CICD_ML_Visualized/) | One offline HTML page, theory only: CI/CD for ML as a visual pipeline (6 step-by-step stories: good change, code bug, bad data, worse model, canary failure, drift → retraining) plus a clickable skeleton of a typical ML repo | `index.html` |
| [`References/`](References/) | 150+ verified links (docs, papers, tutorials, videos) by topic + free cloud comparison | `README.md` |
| `models/` | the Day-2 output shared by Days 3–4: `model_fp32.onnx`, `model_int8.onnx`, `model_card.json`, `labels.json` | — |
| `data/` | datasets, downloaded automatically (git-ignored) | — |

Each day folder has a **`README.md` written for the trainer**: learning objectives, a 5-hour timetable with
lecture / live-coding / break / **student lab** blocks (≈ 35–40 % hands-on), speakable concept notes,
discussion questions, common mistakes, recap and homework.

## 🗓️ The week at a glance

| Day | Theme | Key output | Hands-on |
|---|---|---|---|
| 1 | PyTorch fundamentals | CNN ≈ 91 % on FashionMNIST | ~1 h 45 min labs |
| 2 | Transfer learning + optimization | fp32 90.0 % / int8 89.4 % CIFAR-10, 8.9 MB → 2.6 MB | ~1 h 55 min labs |
| 3 | Flask API + Streamlit | working web app, fp32 vs int8 compared in the service | ~1 h 30 min labs |
| 4 | Docker → Kubernetes → cloud | 2 images, compose, k8s with autoscaling, public URL | ~1 h 15 min labs + follow-along |
| 5 | Thinking like an ML engineer | discussions, debugging, design, quiz | whole day is interactive |

### Where do the live demos go? ("in between" the days)

Each demo is ~60–75 min (training notebook + optimization notebook). Days 1–4 are full, so pick one option:

| Option | GAN demo (`Live_Demos/03` + `04`) | Detection demo (`Live_Demos/01` + `02`) | Cost |
|---|---|---|---|
| **A — recommended** | Day 1, replacing the second half of Lab 2 (3:50–4:30) with the cached GAN run + samples, students finish the lab as homework | Day 2, after the quantization block, in place of part of the Day-2 lab (the detection-optimization notebook *is* a quantization lab on a real model) | labs partly become homework |
| **B** | Opening of Day 5 (replaces the FLEX debate round + part of the "why?" block) | Day 5, right after the GAN demo | Day 5 becomes demo + Q&A (still interactive) |

With `USE_CACHED = True` (the default) every demo notebook loads the pre-trained checkpoint and runs in
seconds to a few minutes, so a demo can never fail live. Set `USE_CACHED = False` to train in front of the class
(detection ≈ 11 min, GAN ≈ 10–19 min on the reference laptop).

## ⚙️ Setup (trainer + students)

Python 3.10–3.13. Students are expected to know how to install Python and create environments.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU build (small); GPU: see pytorch.org
pip install -r requirements.txt
python -m ipykernel install --user --name ml-course --display-name "Python (ml-course)"
```

> ⚠️ **Select the course kernel in Jupyter/VS Code** ("Python (ml-course)" or the `.venv` interpreter).
> On the reference machine the default `python3` kernel pointed to a different (Anaconda) Python with another
> torch version. Every notebook prints `torch.__version__` in its first cell — check it.

Day 4 additionally needs **Docker Desktop** (or Docker Engine) and a local Kubernetes: **minikube** or **kind**
(or Docker Desktop's built-in Kubernetes), plus `kubectl`.

## ✅ Trainer checklist before the course

1. Create the venv, run each day's notebooks once (datasets download on first run: FashionMNIST ~30 MB,
   CIFAR-10 ~170 MB, Penn-Fudan ~54 MB, pretrained weights ~40 MB). For the demos: `python Live_Demos/prepare_demo.py`.
2. Make sure `models/` is populated before Day 3 — if not: `python Day2_Transfer_Learning_and_Optimization/scripts/train_and_export.py` (~6 min on CPU).
3. Day 4: `cd Day4_Docker_Kubernetes_Cloud/ml-app && python sync_from_day3.py && docker compose build` (pre-pulls base images).
4. Deploy the cloud demo **the day before** (Render recommended — see `Day4_Docker_Kubernetes_Cloud/cloud/README.md`),
   and open its URL ~5 minutes before showing it (free services sleep when idle).
5. Skim `References/README.md` → "must-read top 15".
6. Plug the laptop in for live demos (battery mode throttles CPU; timings in the notebooks assume mains power).

## 🔬 What was verified on the reference machine (Windows 11, CPU only, torch 2.9.1)

| Part | Verification | Result |
|---|---|---|
| Day 1 notebooks | executed end-to-end incl. all lab solutions | CNN 91.2 % test, ~3 min |
| Day 2 notebooks + script | executed end-to-end; ONNX checked against PyTorch | fp32 0.900 / ORT int8 0.894, max diff 4e-6 |
| Day 3 API | 13 pytest tests (fake + real model), both notebooks executed | round-trip ~6 ms, ~280 req/s |
| Day 4 Docker | compose build + run, healthchecks, non-root, UI→API by service name, all-in-one image under `--cpus 0.1 --memory 512m` | API image ~110 MB compressed; cold start ~3 min at 0.1 CPU |
| Day 4 Kubernetes | full manifests on kind: probes, self-healing, HPA 2→6 under load, rolling update, broken image + rollback, OOMKilled, readiness exercise | 71,635 requests, 0 errors |
| Live demos | all 4 notebooks executed | detection AP@0.5 0.976; GAN samples recognisable after ~4 epochs |
| References | every link opened on 2026-09-26 | 152 links |

Things you **cannot** verify without your own accounts: the actual Render / Oracle / GCP / Azure deployments
(guides are complete and the images were tested under Render-like limits locally).

## ⚠️ Known caveats (read once)

* **Hugging Face Spaces:** creating Docker Spaces now needs a paid PRO plan → the course uses **Render** for the free public demo.
* **Windows + PyTorch quantization:** the Windows CPU wheel only ships the `onednn` quantization engine (no `fbgemm`/`x86`); the Day-2 notebooks detect and use what's available. `torch.ao.quantization` is deprecated in favour of `torchao` — explained in Day 2.
* **int8 differs slightly across platforms:** the same int8 ONNX file gives slightly different confidences on Windows vs Linux (same top-1). A good Day 4/5 discussion point.
* **`localhost` on Windows** can add ~2 s per request (IPv6 first) — the code uses `127.0.0.1`.
* Free-tier facts were verified on **2026-09-26** and change often — re-check `References/free_cloud_options.md` links the week before.
* `Live_Demos/checkpoints/` holds ~250 MB (detector checkpoint + 2 ONNX exports). They are the live-demo safety net; if you push this repo to GitHub, consider Git LFS for `*.pth`/`*.onnx` (GitHub rejects single files > 100 MB; ours are ~76 MB).
