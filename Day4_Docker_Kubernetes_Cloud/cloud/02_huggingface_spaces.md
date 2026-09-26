# 02 · Hugging Face Spaces (Docker) — only if you have a PRO account

> ⚠️ **Policy change (verified 2026-09-26, [Spaces overview](https://huggingface.co/docs/hub/spaces-overview)):**
> "Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts."
> The CPU Basic hardware itself has no hourly cost, but a **free account can no longer create a Docker Space**.
> Without PRO, use **Render** (`03_render.md`) — it runs the exact same `Dockerfile.allinone`.

**Why (with PRO):** CPU Basic hardware (2 vCPU / 16 GB) at no hourly cost, public HTTPS URL, build logs in
the browser, runs any Dockerfile — nicer hardware than Render's free tier.

**Trade-offs:** one container per Space (so we use `Dockerfile.allinone`: API + UI together), the Space
sleeps after a period of inactivity (first visit afterwards waits for a restart), public by default.

## What gets deployed

```
Space repo (root)
├── Dockerfile      ← copy of ml-app/Dockerfile.allinone
├── start.sh        ← starts gunicorn (API, internal :5000) then Streamlit (public :7860)
├── README.md       ← YAML header: sdk: docker, app_port: 7860
├── api/  ui/  models/
```

## Option A — script (recommended, 5 minutes)

1. Account with a PRO plan: <https://huggingface.co/pricing>.
2. Create a **write** access token: Settings → Access Tokens → *New token* → type **Write**.
3. On your laptop (course venv active), from `Day4_Docker_Kubernetes_Cloud/ml-app/`:
   ```bash
   pip install huggingface_hub
   hf auth login                     # paste the token  (older versions: huggingface-cli login)
   python sync_from_day3.py          # make sure models + samples are in ml-app/
   python deploy_hf_space.py <your-hf-username>/cifar10-demo
   ```
4. Open `https://huggingface.co/spaces/<your-hf-username>/cifar10-demo`.
   The status badge shows **Building** (watch the *Logs* tab — it's the same `docker build` output you saw
   locally) and then **Running**. The app URL is also available as
   `https://<your-hf-username>-cifar10-demo.hf.space`.

To update: change code → `python sync_from_day3.py` → `python deploy_hf_space.py …` again → it rebuilds.

## Option B — web UI only (no local tools)

1. <https://huggingface.co/new-space> → name `cifar10-demo` → **SDK: Docker** → *Blank* → hardware
   **CPU basic (free)** → Create.
2. *Files* → *Add file* → *Upload files*: upload the files keeping the folder structure above
   (rename `Dockerfile.allinone` to `Dockerfile`, and create `README.md` with the header from
   `deploy_hf_space.py` → `SPACE_README`).

## Test the same image locally first

```bash
docker build -f Dockerfile.allinone -t cifar-allinone:v1 .
docker run --rm -p 7860:7860 cifar-allinone:v1
# open http://localhost:7860
```

If it works locally, it will work on the Space — that's the whole point of containers.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Space stuck on "Starting" / app never loads | `app_port: 7860` missing from README header, or Streamlit not on 0.0.0.0:7860 |
| File upload gives `403` | Streamlit XSRF protection behind the proxy — `start.sh` already passes `--server.enableXsrfProtection=false` |
| `permission denied` writing files | Spaces run the container as user **1000** — our image already uses `USER 1000`; write only to `/tmp` or `$HOME` |
| "API not reachable" shown briefly after wake-up | the UI started before the API finished loading; `start.sh` waits for `/ready`, refresh once |
| Build fails at `pip install` | check the *Logs* tab; pin versions if a new release broke something |

## Talking points for class

* The Space is a **managed container platform**: you give it a Dockerfile, it builds, runs, gives HTTPS + a URL.
* No Kubernetes knowledge needed — but also no control over replicas, probes, autoscaling.
* "Free" = shared CPU, sleeping when idle, no SLA. Fine for demos and portfolios, not for production.
