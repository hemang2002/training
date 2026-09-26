# Day 4 — Student Lab: Docker & Kubernetes

Work from `Day4_Docker_Kubernetes_Cloud/ml-app/` (Docker) and `Day4_Docker_Kubernetes_Cloud/` (Kubernetes).
Run `python ml-app/sync_from_day3.py` once first.
Windows PowerShell users: use `curl.exe`; for complex JSON patches, Git Bash is easier.

## Part A — Docker (35 min)

| # | Exercise | Time |
|---|---|---|
| 1 | Build & run the API image, find its size and layers | 8 min |
| 2 | Prove the layer cache works | 7 min |
| 3 | Break it on purpose: `127.0.0.1` vs `0.0.0.0` | 10 min |
| 4 | compose: break and fix service networking | 10 min |
| ★ | Stretch: shrink the UI image | — |

### Exercise 1 — Build, run, inspect
Build `cifar-api:v1`, run it on host port **8000** (not 5000), call `/models`, then answer:
how big is the image? which layer is the largest? what user does the process run as?

<details><summary>✅ Solution</summary>

```bash
docker build -f api/Dockerfile -t cifar-api:v1 .
docker run -d --name api -p 8000:5000 cifar-api:v1      # host:container
curl http://localhost:8000/models
docker images cifar-api
docker history cifar-api:v1        # the venv COPY layer (onnxruntime+numpy) is the largest
docker exec api whoami             # appuser
docker rm -f api
```
`-p 8000:5000` maps host port 8000 to container port 5000 — the app inside still listens on 5000.
</details>

### Exercise 2 — Layer cache
Add a comment line to `api/app.py` and rebuild. Time it. Now add `requests` to `api/requirements.txt` and rebuild.
Which steps say `CACHED`? Why is the second build slower? (Undo both changes afterwards.)

<details><summary>✅ Solution</summary>

Editing `app.py` only invalidates `COPY api/app.py …` and later layers — pip layers stay `CACHED`, build takes seconds.
Editing `requirements.txt` invalidates `COPY api/requirements.txt` → `pip install` runs again → slow.
That is why requirements are copied and installed **before** the code.
</details>

### Exercise 3 — The `0.0.0.0` bug
Run the container overriding the command so gunicorn binds to `127.0.0.1`:
```bash
docker run -d --name bad -p 5001:5000 cifar-api:v1 gunicorn --bind 127.0.0.1:5000 app:app
curl http://localhost:5001/health
```
What happens and why? Prove the app *is* running inside the container.

<details><summary>✅ Solution</summary>

`curl` fails (connection reset / empty reply). Inside the container the app works:
`docker exec bad python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:5000/health').read())"`.
`127.0.0.1` inside a container is the container's own loopback interface; port-mapped traffic arrives on the
container's network interface, so the server must listen on `0.0.0.0`. Clean up: `docker rm -f bad`.
</details>

### Exercise 4 — compose networking
In `docker-compose.yml` change the UI's `API_URL` to `http://localhost:5000`, run `docker compose up -d`, open
http://localhost:8501. Explain the error. Fix it, then use `docker compose exec ui python -c "import socket;print(socket.gethostbyname('api'))"` to see the DNS resolution.

<details><summary>✅ Solution</summary>

Inside the `ui` container `localhost` is the UI container → nothing on port 5000 → "API not reachable".
Compose puts both services on a network with built-in DNS: the hostname `api` resolves to the API container's IP.
Restore `API_URL: http://api:5000` and `docker compose up -d` (compose recreates the changed container).
</details>

### ★ Stretch — shrink the UI image
The UI image contains pandas (used for tables/charts). Measure the image size, then think: which dependencies are
really needed? What would you trade away? (Don't have to implement — discuss size vs developer convenience.)

---

## Part B — Kubernetes (40 min)

Deploy first: `kubectl apply -k k8s/` (see README §5.1–5.2 for loading images).

| # | Exercise | Time |
|---|---|---|
| 5 | Explore: map every YAML file to a live object | 5 min |
| 6 | Break the Service selector | 8 min |
| 7 | Readiness vs liveness in action | 10 min |
| 8 | OOMKilled: set an impossible memory limit | 8 min |
| 9 | Rolling update with zero downtime — measure it | 9 min |
| ★ | Stretch: API key from a Secret | — |

### Exercise 5 — Explore
For each file in `k8s/`, find the live object (`kubectl -n ml-demo get <kind>`), and answer: how many ReplicaSets
exist for `cifar-api` and why? What IPs are behind the `cifar-api` Service?

<details><summary>✅ Solution</summary>

```bash
kubectl -n ml-demo get ns,cm,deploy,rs,pods,svc,hpa
kubectl -n ml-demo get endpoints cifar-api -o wide
```
One ReplicaSet per Deployment *revision* (more appear after rollouts; `revisionHistoryLimit: 3` caps old ones).
The endpoints are the IPs of the **ready** API pods.
</details>

### Exercise 6 — Broken selector
Edit `03-api-service.yaml`: change the selector to `app: cifar-apii`, apply. What does the UI show? What do
`kubectl get endpoints cifar-api` and the pods say? Fix it.

<details><summary>✅ Solution</summary>

Pods are healthy, but the Service selects nothing → `endpoints: <none>` → UI "API not reachable".
No error is raised anywhere — label/selector mismatches are silent. Restore `app: cifar-api` and re-apply.
</details>

### Exercise 7 — Readiness vs liveness
Point the **readiness** probe at a path that doesn't exist (`/nope`) using a patch, and watch the pods:
```bash
kubectl -n ml-demo patch deployment cifar-api --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe/httpGet/path","value":"/nope"}]'
kubectl -n ml-demo get pods -w
```
Are the new pods restarted? Do users see errors? Then do the same for the **liveness** probe and compare.
Undo with `kubectl -n ml-demo rollout undo deployment/cifar-api`.

<details><summary>✅ Solution</summary>

Readiness failing → new pods stay `0/1 READY`, never restarted; the rollout stalls, and because `maxUnavailable: 0`
the **old pods keep serving** → users see no errors.
Liveness failing → the container is killed and restarted repeatedly (RESTARTS count grows → `CrashLoopBackOff`).
Lesson: readiness protects users, liveness recovers stuck processes. Never make liveness depend on things outside
the process (DB, model server) or one outage will restart everything.
</details>

### Exercise 8 — OOMKilled
Set the API memory limit to `40Mi` and watch what happens (`kubectl describe pod`). Restore `512Mi`.

<details><summary>✅ Solution</summary>

```bash
kubectl -n ml-demo set resources deployment cifar-api -c api --limits=memory=40Mi --requests=memory=40Mi
kubectl -n ml-demo get pods -w
kubectl -n ml-demo describe pod <new-pod>    # Last State: Terminated, Reason: OOMKilled, Exit Code: 137
kubectl -n ml-demo set resources deployment cifar-api -c api --limits=memory=512Mi --requests=memory=256Mi
```
Loading ONNX Runtime + two models needs more than 40 MiB. Exit code 137 = 128 + 9 (SIGKILL by the kernel).
</details>

### Exercise 9 — Zero-downtime rollout, measured
Run the load test from your laptop through a port-forward to the **UI-independent** API:
```bash
kubectl -n ml-demo port-forward svc/cifar-api 5000:5000      # terminal 1
python k8s/loadtest/loadtest.py --url http://localhost:5000 --duration 90 --concurrency 4   # terminal 2
kubectl -n ml-demo rollout restart deployment/cifar-api     # terminal 3, while the test runs
```
Did you get errors? Why might port-forward itself cause errors here (hint: it connects to *one* pod)?
Repeat using the in-cluster Job (README §5.4) and compare.

<details><summary>✅ Solution</summary>

With the in-cluster Job going through the Service, the rolling update should produce **0 (or very few) errors**:
new pods only receive traffic when ready; old pods are removed from endpoints before termination.
`kubectl port-forward` binds to a single pod chosen at start — when that pod is terminated during the rollout,
the port-forward breaks and every request errors. That's a tooling artifact, not an app failure. Lesson: test through
the same path your users use.
</details>

### ★ Stretch — API key from a Secret
Implement Day 3 LAB exercise 2 (API key), rebuild as `cifar-api:v2`, load it into the cluster, create a Secret
(`kubectl -n ml-demo create secret generic cifar-secrets --from-literal=API_KEY=s3cret`), inject it with
`secretKeyRef` (see `k8s/extras/secret-example.yaml`), pass it to the UI the same way, and roll out v2.
