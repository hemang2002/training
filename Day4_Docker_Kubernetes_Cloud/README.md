# Day 4 — Ship It: Docker → Kubernetes → Free Cloud

> **Theme of the day:** *"It works on my machine."* — Fine, then we ship the machine.
> We package the Day 3 API and UI into containers, run them with docker-compose, orchestrate them
> with Kubernetes (replicas, probes, rolling updates, autoscaling), and finally put the app on a
> free cloud tier so students can open it on their phones.

> 🎓 **Teaching visually?** Open [`../Docker_K8s_Visualized/index.html`](../Docker_K8s_Visualized/index.html): one interactive page per concept
> (container → image → compose → Pod → Deployment → Service → ConfigMap → probes → HPA), each with an "Explain it like this" box,
> plus a step-by-step lab with the real output of every command.

## 🎯 Learning objectives

1. Explain images vs containers vs registries, layers and the build cache.
2. Write production-style Dockerfiles (slim base, layer order, multi-stage, non-root, healthcheck) and **measure** the difference vs a naive one.
3. Run a multi-container app with docker-compose and explain container networking (service names vs `localhost`).
4. Explain the core Kubernetes objects — Pod, Deployment, ReplicaSet, Service, ConfigMap, Secret, HPA — and why each exists.
5. Deploy the app to a local cluster; demonstrate self-healing, load balancing, rolling update, rollback and autoscaling.
6. Deploy the app to a free cloud platform and understand what "free tier" trades away (sleep, cold starts, limits).

## 📁 What's in this folder

```
Day4_Docker_Kubernetes_Cloud/
├── README.md                       ← this guide (lecture notes + step-by-step commands)
├── LAB.md                          ← student exercises + solutions
├── ml-app/                         ← the self-contained deployable unit (= Docker build context)
│   ├── sync_from_day3.py           ← copies latest code + samples + models from Day 3 / models/
│   ├── api/  Dockerfile  Dockerfile.naive  app.py  model_service.py  requirements.txt
│   ├── ui/   Dockerfile  streamlit_app.py  requirements.txt  samples/
│   ├── models/                     ← model_fp32.onnx, model_int8.onnx, model_card.json, labels.json
│   ├── docker-compose.yml          ← api + ui together
│   ├── Dockerfile.allinone         ← api + ui in ONE container (for Render / Hugging Face Spaces)
│   ├── start.sh                    ← entry point of the all-in-one image
│   └── .dockerignore
├── k8s/
│   ├── 00-namespace.yaml … 06-api-hpa.yaml, kustomization.yaml
│   ├── loadtest/ loadtest.py, loadtest-job.yaml   ← generate traffic to watch scaling
│   └── extras/   ingress.yaml, secret-example.yaml
└── cloud/
    ├── README.md                   ← which free platform to use (comparison + recommendation)
    ├── 01_docker_hub.md            ← push images to a registry (needed by most options)
    ├── 02_huggingface_spaces.md    ← only with a PRO plan (free accounts can't create Docker Spaces)
    ├── 03_render.md                ← ⭐ free public URL from our Docker image, no card
    ├── 04_vm_docker_compose.md     ← any free VM (Oracle / GCP / Azure / AWS): docker compose
    ├── 05_vm_k3s_kubernetes.md     ← real Kubernetes on a free VM with k3s + our manifests
    └── 06_killercoda_playground.md ← free browser Kubernetes, no install, no account card
```

## 🕐 Timetable (5 hours)

| Time | Block | Type |
|------|-------|------|
| 0:00 – 0:10 | Recap Day 3; "works on my machine" story | Discussion |
| 0:10 – 0:45 | **Docker concepts**: VM vs container, image/container/registry, layers & cache, Dockerfile instructions | Lecture (§1) |
| 0:45 – 1:25 | **Live**: build naive vs good API image, compare size, run, logs, exec, healthcheck; UI image; compose | Live coding (§2–§3) |
| 1:25 – 1:40 | ☕ Break | |
| 1:40 – 2:15 | 🧑‍💻 **Student lab A — Docker** (LAB.md 1–4) | Hands-on |
| 2:15 – 2:50 | **Kubernetes concepts**: why orchestration, control plane, objects, labels/selectors, probes, resources | Lecture (§4) |
| 2:50 – 3:30 | **Live**: deploy to minikube; self-healing, scaling, load-balancing, rolling update + rollback, HPA with load test | Live coding (§5) |
| 3:30 – 3:45 | ☕ Break | |
| 3:45 – 4:25 | 🧑‍💻 **Student lab B — Kubernetes** (LAB.md 5–9) | Hands-on |
| 4:25 – 4:55 | **Cloud**: free-tier options, Render deployment (prepared beforehand), tour of VM + k3s | Demo (§6) |
| 4:55 – 5:00 | Recap + Day 5 preview | Wrap-up |

Hands-on ≈ 1 h 15 min + students follow the live commands on their own machines.
**If short on time:** skip the naive-image build (show the numbers from the table below) and the ingress extra.
**If ahead:** LAB stretch exercises (Secrets, resource limits → OOMKilled, readiness failure).

---

## ✅ Trainer preparation (the day before)

1. Docker Desktop running (Windows/macOS) or Docker Engine (Linux). `docker run hello-world` works.
2. A local cluster: **one** of
   * **minikube** (recommended, used below): `minikube start --cpus 4 --memory 6g`
   * **kind**: `kind create cluster --name ml`
   * **Docker Desktop → Settings → Kubernetes → Enable**
3. `kubectl version --client` works.
4. Models exist in `<repo>/models/` (Day 2). Then:
   ```bash
   cd Day4_Docker_Kubernetes_Cloud/ml-app
   python sync_from_day3.py
   docker compose build          # pre-pull base images so the live build is fast
   ```
5. Deploy to Render **before class** (`cloud/03_render.md`) and open the URL ~5 min before showing it (free services sleep after 15 min idle; our container needs ~3 min to wake on 0.1 CPU).
6. Pre-pull on venue Wi-Fi is risky: images `python:3.12-slim` and `busybox` should already be cached.

---

## 1. Docker concepts (lecture notes)

### 1.1 Why containers?

The same code behaves differently across machines because of **different Python versions, library
versions, OS packages, environment variables**. A container image freezes all of that.

```
 Virtual machine                         Container
┌───────────────────────┐        ┌───────────────────────┐
│ App A │ App B         │        │ App A │ App B         │
│ libs  │ libs          │        │ libs  │ libs          │
│ Guest OS │ Guest OS   │        ├───────────────────────┤
├───────────────────────┤        │ Container runtime     │
│ Hypervisor            │        │ Host OS kernel (shared)│
│ Host OS / hardware    │        │ Hardware              │
└───────────────────────┘        └───────────────────────┘
 GBs, boots in minutes             MBs, starts in seconds
```

Containers are **isolated processes** (Linux namespaces + cgroups) sharing the host kernel. On Windows/macOS,
Docker Desktop runs a small Linux VM behind the scenes (WSL 2 on Windows).

### 1.2 Vocabulary

| Term | Analogy | Example |
|---|---|---|
| **Dockerfile** | recipe | `api/Dockerfile` |
| **Image** | frozen, read-only template (stack of layers) | `cifar-api:v1` |
| **Container** | a running instance of an image | `docker run cifar-api:v1` |
| **Registry** | app store for images | Docker Hub, GHCR, GCR/Artifact Registry, ECR |
| **Tag** | version label | `:v1`, `:v2`, avoid relying on `:latest` |
| **Volume** | storage outside the container's writable layer | model cache, DB data |

### 1.3 Layers and the build cache — the single most useful Docker idea

Every instruction creates a layer. Docker re-uses a cached layer if the instruction **and everything before it**
is unchanged. So order instructions from *least* to *most* frequently changing:

```dockerfile
COPY api/requirements.txt .       # changes rarely
RUN pip install -r requirements.txt   # slow -> cached most of the time
COPY api/app.py ./                # changes often -> only this layer rebuilds
```

Put `COPY . .` before `pip install` and **every** code edit reinstalls all packages.

### 1.4 Walk through `ml-app/api/Dockerfile`

| Line(s) | Why |
|---|---|
| `FROM python:3.12-slim AS builder` | slim base ≈ 130 MB vs ≈ 1 GB for `python:3.12` |
| multi-stage: venv built in `builder`, copied into runtime | build leftovers (pip cache, build tools) never reach the final image |
| `COPY requirements.txt` → `pip install` → `COPY app.py` | layer cache (see above) |
| `ENV MODEL_DIR=… DEFAULT_MODEL=… WORKERS=…` | configuration via environment (12-factor) — overridable at run time |
| `useradd` + `USER 1000` | never run as root; numeric UID so k8s `runAsNonRoot` can verify it |
| `HEALTHCHECK` | `docker ps` shows `healthy`; compose uses it for `depends_on` |
| `exec gunicorn … 0.0.0.0` | `exec` → PID 1 gets SIGTERM → graceful stop; `0.0.0.0` so traffic from outside the container is accepted |

### 1.5 `Dockerfile.naive` — the anti-example

Read its header comment with the students and let them find the 7 problems before you reveal them.

---

## 2. Docker hands-on (live commands)

All commands from `Day4_Docker_Kubernetes_Cloud/ml-app/`.

```bash
python sync_from_day3.py                              # code + samples + models into the build context

# --- build ---
docker build -f api/Dockerfile -t cifar-api:v1 .
docker build -f api/Dockerfile.naive -t cifar-api:naive .   # optional, slow: downloads torch
docker images | grep cifar                             # PowerShell: docker images | findstr cifar
docker history cifar-api:v1                            # the layers and their sizes

# --- run ---
docker run -d --name api -p 5000:5000 cifar-api:v1
docker ps                                              # STATUS becomes "(healthy)" after ~30 s
docker logs -f api                                     # Ctrl+C to stop following
curl http://localhost:5000/ready
curl -F "file=@ui/samples/cat_1.png" http://localhost:5000/predict

# --- look inside ---
docker exec -it api sh                                 # whoami -> appuser ; ls /app/models ; exit
docker stats api                                       # live CPU / memory

# --- configuration without rebuilding ---
docker rm -f api
docker run -d --name api -p 5000:5000 -e DEFAULT_MODEL=fp32 -e WORKERS=1 cifar-api:v1
curl http://localhost:5000/models                      # default is now fp32

# --- clean up ---
docker rm -f api
```

> Windows PowerShell: use `curl.exe` instead of `curl`.

### "What difference does it make?" — image size

Fill in live with `docker images` (exact numbers vary by version/platform):

| Image | Base | Contains torch? | Size | Rebuild after editing `app.py` |
|---|---|---|---|---|
| `cifar-api:naive` | `python:3.12` (~1 GB) | yes — on Linux, plain `pip install torch` pulls the **CUDA** build + NVIDIA libs | **several GB** (typically 6–9 GB) | full reinstall (many minutes) |
| `cifar-api:v1` | `python:3.12-slim`, multi-stage | no (ONNX Runtime) | **measured: ~110 MB compressed** (what a registry push/pull transfers); `docker images` on Docker Desktop shows ~450 MB on disk; venv inside = 181 MB | seconds (cached pip layer) |

Other measured images: `cifar-ui:v1` ~167 MB compressed (Streamlit + pandas are the bulk), `cifar-allinone:v1` ≈ both together.
Measured running memory (`docker stats`): API ~150 MiB, UI ~50 MiB.

Even a "fixed" torch image using CPU-only wheels (`--index-url https://download.pytorch.org/whl/cpu`) is
still ~1 GB — the runtime choice (ONNX Runtime) is the biggest win.

Consequences: faster pushes/pulls, faster pod start-up and autoscaling, fits free tiers, smaller attack surface.

## 3. docker-compose (live)

```bash
docker compose up --build            # add -d to run in background
# UI:  http://localhost:8501      API: http://localhost:5000/health
docker compose ps
docker compose logs -f ui
docker compose down
```

Key teaching point — **networking**: compose creates a private network where each service is reachable by
its **service name**. Inside the `ui` container, `localhost` is the UI container itself, so
`API_URL=http://api:5000` (see `docker-compose.yml`). Demonstrate the bug: set `API_URL: http://localhost:5000`,
`docker compose up`, watch the UI sidebar show "API not reachable", fix it.

`depends_on: condition: service_healthy` → the UI starts only after the API's HEALTHCHECK passes.

---

## 4. Kubernetes concepts (lecture notes)

### 4.1 Why orchestration?

Compose runs containers on **one** machine. In production we need: many machines, restart on crash,
replace failed machines, spread traffic, scale with load, deploy new versions without downtime, keep config
and secrets out of images. That's Kubernetes.

**Declarative model:** you write *desired state* in YAML; controllers continuously make *actual state* match.

```
                 kubectl apply -f  (desired state)
                          │
               ┌──────────▼───────────┐   Control plane
               │ API server ── etcd    │   (stores state)
               │ scheduler             │   (picks a node for each pod)
               │ controller-manager    │   (reconcile loops: Deployment, ReplicaSet, HPA…)
               └──────────┬───────────┘
          ┌───────────────┼────────────────┐
     ┌────▼────┐     ┌────▼────┐      ┌────▼────┐   Worker nodes
     │ kubelet │     │ kubelet │      │ kubelet │   (start containers, run probes)
     │ pod pod │     │ pod     │      │ pod pod │
     └─────────┘     └─────────┘      └─────────┘
```

### 4.2 The objects we use (map each to a file)

| Object | What it does | Our file |
|---|---|---|
| **Namespace** | groups resources, easy cleanup | `00-namespace.yaml` |
| **ConfigMap** | non-secret config as env vars/files | `01-configmap.yaml` |
| **Deployment** → ReplicaSet → **Pods** | desired # of identical pods, rolling updates, rollback | `02-api-deployment.yaml`, `04-ui-deployment.yaml` |
| **Service** | stable name + virtual IP, load-balances over *ready* pods selected by label | `03-api-service.yaml` (ClusterIP), `05-ui-service.yaml` (NodePort) |
| **HorizontalPodAutoscaler** | changes replicas based on CPU | `06-api-hpa.yaml` |
| Secret | sensitive config (base64, not encrypted by default!) | `extras/secret-example.yaml` |
| Ingress | HTTP routing by host/path through one entry point | `extras/ingress.yaml` |
| Job | run-to-completion task | `loadtest/loadtest-job.yaml` |

**Labels & selectors** glue everything together: the Deployment's `selector`, the pod template `labels`,
and the Service `selector` must all agree on `app: cifar-api`. A typo here = Service with no endpoints.

### 4.3 Probes (connect to Day 3's `/health` and `/ready`)

| Probe | Question | On failure |
|---|---|---|
| `startupProbe` | has the app finished starting? (model loading) | other probes wait; after `failureThreshold × period` → restart |
| `readinessProbe` → `/ready` | can it take traffic now? | removed from Service endpoints (no restart) |
| `livenessProbe` → `/health` | is it stuck/dead? | container restarted |

### 4.4 Requests & limits

* **requests** — reserved for scheduling; HPA utilization % is relative to the CPU *request*.
* **limits** — CPU above limit is *throttled*; memory above limit → container **OOMKilled**.
* Why `WORKERS=1` in the ConfigMap: in k8s, scale with **pods**, not with processes inside a pod —
  the scheduler and HPA can only see pods.

### 4.5 Service types

| Type | Reachable from | Use |
|---|---|---|
| ClusterIP | inside cluster only | API (internal) |
| NodePort | `<any-node-ip>:30000-32767` | demos, bare VMs (our UI) |
| LoadBalancer | external IP from the cloud provider | managed clusters (costs money on most clouds) |

---

## 5. Kubernetes hands-on with minikube (live commands)

All commands from `Day4_Docker_Kubernetes_Cloud/`.

### 5.1 Start the cluster and load the images

```bash
minikube start --cpus 4 --memory 6g
minikube addons enable metrics-server         # needed by the HPA
kubectl get nodes

# The cluster runs its own container runtime - it cannot see images on your laptop's Docker.
# Either load them ...
minikube image load cifar-api:v1
minikube image load cifar-ui:v1
# ... or build directly inside minikube:  (cd ml-app && minikube image build -t cifar-api:v1 -f api/Dockerfile .)
minikube image ls | grep cifar
```

<details><summary>Using kind or Docker Desktop instead?</summary>

* **kind:** `kind create cluster --name ml` → `kind load docker-image cifar-api:v1 cifar-ui:v1 --name ml`.
  Metrics server: `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`
  then `kubectl -n kube-system patch deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'`.
  Open the UI with `kubectl -n ml-demo port-forward svc/cifar-ui 8501:8501`.
* **Docker Desktop Kubernetes:** images built with your local Docker are usually visible directly (no load step).
  Install metrics-server as for kind. UI at `http://localhost:30080`.
</details>

### 5.2 Deploy

```bash
kubectl apply -k k8s/                          # applies everything listed in kustomization.yaml
kubectl -n ml-demo get all
kubectl -n ml-demo get pods -w                 # watch: ContainerCreating -> Running, READY 0/1 -> 1/1
kubectl -n ml-demo describe pod -l app=cifar-api   # events at the bottom: scheduling, pulling, probes
kubectl -n ml-demo logs -l app=cifar-api --tail=20
kubectl -n ml-demo get endpoints cifar-api     # the pod IPs behind the Service

minikube service cifar-ui -n ml-demo --url     # open this URL in the browser
```

Check the pod name shown under the prediction in the UI ("Served by pod") — click Predict a few times.

### 5.3 Demo: self-healing

```bash
kubectl -n ml-demo get pods -l app=cifar-api
kubectl -n ml-demo delete pod <one-api-pod-name>
kubectl -n ml-demo get pods -l app=cifar-api -w     # a replacement appears immediately
```

*Ask:* who created the new pod? (the ReplicaSet controller — desired 2, actual 1 → create one).

### 5.4 Demo: scaling & load balancing

```bash
kubectl -n ml-demo scale deployment cifar-api --replicas=4
# the HPA will scale it back to its own target later - that's expected, discuss it!
kubectl -n ml-demo create configmap loadtest-script --from-file=k8s/loadtest/loadtest.py
kubectl apply -f k8s/loadtest/loadtest-job.yaml
kubectl -n ml-demo logs -f job/loadtest          # at the end: requests per pod -> evenly spread
```

### 5.5 Demo: autoscaling (HPA)

```bash
# terminal 1
kubectl -n ml-demo get hpa cifar-api -w
# terminal 2 (re-run the load test)
kubectl -n ml-demo delete job loadtest --ignore-not-found
kubectl apply -f k8s/loadtest/loadtest-job.yaml
kubectl -n ml-demo get pods -w
```

Expect TARGETS to climb above 60% and REPLICAS to grow (max 6) within ~1 minute; after the job ends,
it scales back to 2 after the 60 s stabilization window. If TARGETS shows `<unknown>`, metrics-server isn't ready yet — wait 1–2 min.

### 5.6 Demo: rolling update and rollback

```bash
# Change configuration: make fp32 the default and roll the pods
kubectl -n ml-demo patch configmap cifar-config -p '{"data":{"DEFAULT_MODEL":"fp32"}}'
kubectl -n ml-demo rollout restart deployment/cifar-api
kubectl -n ml-demo rollout status deployment/cifar-api
kubectl -n ml-demo rollout history deployment/cifar-api

# Ship a "broken" version: an image tag that doesn't exist
kubectl -n ml-demo set image deployment/cifar-api api=cifar-api:v2-typo
kubectl -n ml-demo get pods                        # new pod: ErrImagePull / ImagePullBackOff
                                                   # old pods keep serving (maxUnavailable: 0) -> no downtime!
kubectl -n ml-demo rollout undo deployment/cifar-api
kubectl -n ml-demo rollout status deployment/cifar-api
```

> PowerShell quoting: use `-p '{\"data\":{\"DEFAULT_MODEL\":\"fp32\"}}'` or run the patch from Git Bash.

`rollout undo` prints *"Warning: resource deployments/cifar-api was previously managed with 'kubectl apply'…"*.
That's expected: imperative commands (`set image`, `undo`, `patch`) change the live object but not the YAML in
git. Discussion point → **GitOps**: the YAML in the repo should be the source of truth; after an emergency
rollback, fix the file and `kubectl apply -k k8s/` again.

**Verified behaviour (kind v0.27, this repo's manifests):** ConfigMap change + `rollout restart` switched the default
model to fp32; the broken tag gave `ImagePullBackOff` on 1 new pod while the old pods kept answering `/ready` with 200;
`rollout undo` restored service. In the HPA demo, the load-test Job pushed CPU to 360% of request → 2 → 6 replicas
in about 1 minute, 71,635 requests, **0 errors**, spread over all 6 pods.

### 5.7 Clean up

```bash
kubectl delete -k k8s/             # or: kubectl delete namespace ml-demo
minikube stop                      # minikube delete  to remove the cluster entirely
```

---

## 6. Deploying to a free cloud (demo)

Start with `cloud/README.md` — it compares the free options and recommends:

| Goal | Use | Guide |
|---|---|---|
| Show the app on a public URL, **no credit card** | Render free web service (all-in-one image) | `cloud/03_render.md` |
| Same, if you own a Hugging Face **PRO** plan | Hugging Face Docker Space | `cloud/02_huggingface_spaces.md` |
| "Real server" experience: SSH, docker compose on a VM | Oracle Always Free / GCP / Azure for Students / AWS credits | `cloud/04_vm_docker_compose.md` |
| **Real Kubernetes** with our exact manifests | k3s on a free VM | `cloud/05_vm_k3s_kubernetes.md` |
| Kubernetes in the browser, nothing to install | Killercoda playground | `cloud/06_killercoda_playground.md` |

Suggested class demo (25 min): open the pre-deployed Render service on the projector and let students hit it
from their phones; show its build logs; walk through the k3s guide and explain how the **same YAML** from
§5 runs unchanged on a cloud VM — only the image names change (Docker Hub instead of local images).

**Honesty about free tiers** (discuss): apps sleep when idle → first request is slow (cold start); small CPU/RAM;
no SLA; limits and offers change — always re-check the official page before class.

---

## ⚠️ Common problems (and what they teach)

| Symptom | Cause | Fix |
|---|---|---|
| `failed to connect to the docker API` | Docker Desktop not running | start it; wait for "Engine running" |
| UI: "API not reachable" in compose | `API_URL=http://localhost:5000` inside a container | use the service name `http://api:5000` |
| `curl: (52) Empty reply` / connection refused from host | app listens on `127.0.0.1` inside the container | bind `0.0.0.0` |
| `ErrImagePull` / `ImagePullBackOff` | image not loaded into minikube/kind, or wrong tag/registry | `minikube image load …`, check `kubectl describe pod` |
| `CreateContainerConfigError: … non-numeric user` | `runAsNonRoot` with `USER appuser` | `USER 1000` (we already do this) |
| `CrashLoopBackOff` | app crashes at start (missing models, bad env) | `kubectl logs <pod> --previous` |
| `OOMKilled` in `describe pod` | memory limit too low (e.g. many gunicorn workers) | raise limit or lower `WORKERS` |
| Pod `Running` but `READY 0/1` | readiness probe failing | `describe pod` → Events; check `/ready` |
| Service works but some requests fail | selector/label mismatch or a pod not ready | `kubectl get endpoints cifar-api` |
| HPA `TARGETS <unknown>` | metrics-server missing / starting | enable addon, wait |
| `exec ./start.sh: no such file` in all-in-one image | CRLF line endings | we strip them with `sed` in the Dockerfile; add `.gitattributes` |
| Apple Silicon / ARM VM: `exec format error` | image built for amd64 only | `docker buildx build --platform linux/amd64,linux/arm64 … --push` |

## 💬 Discussion questions

1. Why is a 300 MB image better than a 3 GB image *for autoscaling* specifically?
2. We bake the model into the image. What are the alternatives (volume, download at start from object storage, model registry)? Trade-offs?
3. What exactly happens to in-flight requests when a pod is deleted during a rolling update? (SIGTERM, endpoints removal, graceful timeout)
4. Why is the HPA target based on CPU *request* and not on the limit?
5. Would you autoscale this API on CPU, on request rate, or on latency? Why?
6. Why is putting API + UI in one container OK for Render's free tier but not for our Kubernetes deployment?
7. We measured the **same int8 ONNX file** with the same ONNX Runtime version on Windows (laptop) and Linux
   (container): fp32 outputs were identical, int8 confidences differed slightly (e.g. 0.850 vs 0.840 for `cat_1`),
   top-1 accuracy on the samples was the same. Why? (different optimized integer kernels per OS/CPU path)
   What does that mean for where you validate a quantized model? (on the deployment platform, not only the laptop)
8. Base64 Secrets are not encrypted. How do real teams manage secrets? (encryption at rest, external secret managers, sealed secrets)

## 📝 Recap & homework

**Recap:** Dockerfile best practices (and measured impact) → compose networking → k8s declarative objects →
probes/resources/rollouts/HPA → same artifacts on a free cloud.

**Homework:** deploy your own copy to Render and send the URL; finish LAB exercises;
prepare one question about anything from the week for Day 5.
