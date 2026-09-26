# 03 · Render — ⭐ recommended free public deployment

**Why:** classic PaaS experience (connect repo or image → get an HTTPS URL), free instance type without
entering a card, runs our Dockerfile, good dashboard (logs, events, metrics).

**Trade-offs (verified 2026-09-26, <https://render.com/docs/free>):** free web services have 512 MB RAM and
0.1 CPU, 750 free instance-hours per workspace per month, and **spin down after 15 minutes without traffic** —
Render quotes ~1 minute to spin up, but **our all-in-one container needs ~2.5–3 minutes on 0.1 CPU**: measured locally with `docker run --cpus 0.1 --memory 512m` → models load in ~25 s, Streamlit's import takes ~2 min, memory ~140 MiB. **Open the URL ~5 minutes before showing it in class.**
With 0.1 CPU, inference is noticeably slower than on a laptop — a great talking point about what "free" means.

We deploy the **all-in-one** image (one web service = one container). Render injects a `PORT` environment
variable; `start.sh` runs Streamlit on `${PORT}` automatically.

## Option A — from Docker Hub (fastest)

1. Push the all-in-one image (see `01_docker_hub.md`):
   ```bash
   docker build -f Dockerfile.allinone -t <dockerhub-user>/cifar-allinone:v1 .
   docker push <dockerhub-user>/cifar-allinone:v1
   ```
2. <https://dashboard.render.com> → **New** → **Web Service** → **Existing image** →
   `docker.io/<dockerhub-user>/cifar-allinone:v1`.
3. Instance type: **Free**. Region: nearest. Create.
4. Watch the *Logs* tab: "Waiting for API... API ready" then Streamlit's "You can now view your app".
5. Open `https://<service-name>.onrender.com`.

## Option B — from a GitHub repo (auto-deploy on every push)

1. Push this course repo (or just `ml-app/` incl. `models/`) to GitHub.
2. Render → **New** → **Web Service** → connect the repo.
3. Settings:
   * **Language / Runtime:** Docker
   * **Root Directory** / **Docker Build Context Directory:** `Day4_Docker_Kubernetes_Cloud/ml-app`
   * **Dockerfile Path:** `Day4_Docker_Kubernetes_Cloud/ml-app/Dockerfile.allinone`
   * **Instance type:** Free
4. Every `git push` now triggers build + deploy (a mini CI/CD pipeline — good discussion point).

## Health check (optional)

Settings → Health Check Path: `/_stcore/health` (Streamlit's endpoint). Render then only switches traffic to a
new deploy once it is healthy — the same idea as a Kubernetes readiness probe.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No open ports detected" | Streamlit must listen on `0.0.0.0:$PORT` — `start.sh` does this |
| Service restarts with out-of-memory | free RAM is small: keep `--workers 1` in `start.sh`; serve only the int8 model if needed |
| Very slow first request | cold start after spin-down — expected on the free tier; mention it in class |
| `exec format error` | image built for the wrong architecture — Render uses linux/amd64 |
