# 06 · Killercoda — Docker + Kubernetes in the browser (nothing to install on your laptop)

**What it is:** a free Linux computer in a browser tab. We use Docker there to build our app, and a small
Kubernetes (**k3s**) that runs on that same Docker. **No Docker Hub, no laptop setup.**

```
            one Killercoda Ubuntu machine
 ┌──────────────────────────────────────────────────┐
 │  GitHub ──git clone──▶ course files                │
 │                          │ docker build            │
 │                          ▼                         │
 │  Docker  ◀── images cifar-api:v1, cifar-ui:v1      │
 │    ▲                                               │
 │    └── k3s (Kubernetes) runs its pods on this Docker│
 │         → it sees our images directly              │
 └──────────────────────────────────────────────────┘
```

You need: a free **Killercoda** account (sign in with GitHub/Google) and the course on **GitHub** (public repository).
Total time: ~10 minutes. Everything is copy-paste.

> ⚠️ Use the **Ubuntu** playground, not the *Kubernetes* playground — the Kubernetes playground has no Docker.

---

## Step 1 — Open the playground

Open **https://killercoda.com/playgrounds/scenario/ubuntu** → sign in → **Start** →
wait until the terminal on the right shows a prompt (`ubuntu:~$` or `root@...`).

## Step 2 — Docker: check it (install only if missing)

```bash
docker version
```
* Shows **Client** and **Server** → Docker is ready, go to Step 3.
* Says `command not found` → install it (≈1 min):
  ```bash
  curl -fsSL https://get.docker.com | sh
  ```

## Step 3 — Install Kubernetes (k3s) on top of Docker

```bash
curl -sfL https://get.k3s.io | sh -s - --docker
```
`--docker` = *"use the Docker that is already here"*. That's why we need no Docker Hub.
k3s also installs `kubectl` for you.

### ✅ Check
```bash
kubectl get nodes
```
One node with STATUS **Ready**. (Says `NotReady` or an error? Wait 30 s and run it again.)

## Step 4 — Get the course from GitHub

```bash
git clone --depth 1 https://github.com/hemang2002/training.git
cd training/Day4_Docker_Kubernetes_Cloud
```

## Step 5 — Build the app images with Docker (≈3–5 min the first time)

```bash
docker build -t cifar-api:v1 -f ml-app/api/Dockerfile ml-app
docker build -t cifar-ui:v1  -f ml-app/ui/Dockerfile  ml-app
```

### ✅ Check
```bash
docker images | grep cifar
```
Two lines: `cifar-api  v1` and `cifar-ui  v1`.

## Step 6 — Run the app on Kubernetes

```bash
kubectl apply -k k8s/
kubectl -n ml-demo get pods -w
```
Wait until all 3 pods show `Running` and READY `1/1` (about 1 minute), then press **Ctrl + C**.

### ✅ Check — open the app
Killercoda menu (☰, top right) → **Traffic / Ports** → type **30080** → **Access**.
The app opens in a new tab → pick a picture → prediction. 🎉

---

## Try it: Kubernetes in action (5 min)

**Self-healing** — delete both API copies, Kubernetes makes new ones:
```bash
kubectl -n ml-demo delete pod -l app=cifar-api --wait=false
kubectl -n ml-demo get pods -w
```

**Kubernetes really uses Docker here** — the pods are ordinary Docker containers:
```bash
docker ps --format "{{.Names}}" | grep cifar
```

More demos (scaling, rolling update, rollback) → main README §5.4 and §5.6 — the same commands work here.

Nothing to clean up: the whole machine is deleted when the session ends.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `git clone` asks for **Username / Password** | the GitHub repository is private → on GitHub: repository → **Settings** → **Change visibility** → **Public** |
| `kubectl get nodes` → `connection refused` | k3s is still starting — wait 30 s, try again |
| `docker build` fails with `no space left on device` | start a new session (the disk is small); run the steps again |
| Pods stay `Pending` | `kubectl -n ml-demo describe pod <name>` → bottom lines say why (usually CPU/memory: wait, or delete the HPA with `kubectl -n ml-demo delete hpa cifar-api`) |
| Pods `ErrImageNeverPull` / `ErrImagePull` | the images are missing or k3s was installed **without** `--docker` → do Step 5 again; check Step 3 used `--docker` |
| Traffic page shows an error | wait until `cifar-ui` is READY `1/1`, reload |
