# Install Docker + Kubernetes and run the app — Windows & Linux

At the end you will have our image-classifier app running **twice** on your own computer:

1. with **Docker** (docker compose) → http://localhost:8501
2. with **Kubernetes** → http://localhost:30080

You only install **one program: Docker Desktop**. Kubernetes is a switch inside it.

| Part | What | Windows | Linux (Ubuntu) |
|---|---|---|---|
| 1 | Install Docker Desktop | ~10 min | ~15 min |
| 2 | Run the app with Docker | 5 min | 5 min |
| 3 | Turn on Kubernetes | 3 min | 5 min (+ install `kubectl`) |
| 4 | Run the app on Kubernetes | 5 min | 5 min |

---

## Part 1 — Install Docker Desktop

### 🪟 Windows 10 / 11

1. Go to **https://www.docker.com/products/docker-desktop/** → **Download for Windows**.
2. Run `Docker Desktop Installer.exe`. Keep **"Use WSL 2 instead of Hyper-V"** ticked → **OK** → **Close and restart**.
3. After the restart, open **Docker Desktop** from the Start menu, accept the terms (you can skip sign-in).
4. Wait until the bottom-left corner says **"Engine running"**.

> If Docker asks you to *update WSL*, click the button it shows, then open Docker Desktop again.

### 🐧 Linux (Ubuntu 24.04 / 26.04, 64-bit, with a desktop)

Docker Desktop for Linux needs **KVM** (hardware virtualization). Open a terminal:

**1. Check KVM and give yourself access**
```bash
ls -al /dev/kvm                    # must exist; if not, turn on virtualization (VT-x / AMD-V) in the BIOS
sudo usermod -aG kvm $USER         # then LOG OUT and LOG IN again
```

**2. Add Docker's package repository** (copy the whole block)
```bash
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
```

**3. Download and install Docker Desktop**
```bash
curl -LO https://desktop.docker.com/linux/main/amd64/docker-desktop-amd64.deb
sudo apt install ./docker-desktop-amd64.deb
```
(A message *"Download is performed unsandboxed as root…"* at the end is normal — ignore it.)

**4. Start it:** open **Docker Desktop** from your applications menu, accept the terms, wait for **"Engine running"**.

### ✅ Check (Windows and Linux)

```bash
docker
```

---

## Part 2 — Run the app with Docker

Open a terminal **in the course folder** (the one that contains `Day4_Docker_Kubernetes_Cloud`):
* Windows: open the folder in File Explorer → click the address bar → type `powershell` → Enter
* Linux: right-click in the folder → **Open in Terminal**

```bash
cd Day4_Docker_Kubernetes_Cloud/ml-app
docker compose up --build
```

The first time takes a few minutes (Docker downloads Python and the libraries). Wait until you see
`api-1 | ... Listening at: http://0.0.0.0:5000` and `ui-1 | ... You can now view your Streamlit app`.

### ✅ Check
Open **http://localhost:8501** → pick a sample picture → the model says what it is. 🎉

Stop with **Ctrl + C** in the terminal. (The images you just built, `cifar-api:v1` and `cifar-ui:v1`, stay on
your computer — Kubernetes will use them in Part 4.)

---

## Part 3 — Turn on Kubernetes

### 🪟 Windows and 🐧 Linux — same clicks

1. Docker Desktop → left menu **Kubernetes** → **Create cluster**
   (older versions: ⚙️ **Settings → Kubernetes → Enable Kubernetes → Apply**).
2. Choose **Kubeadm** (single node — simplest, and it can see the images you built in Part 2) → **Create**.
3. Wait ~1–3 minutes until it shows **Kubernetes running** (green).

### 🐧 Linux only — install `kubectl`
Docker Desktop for Windows already includes `kubectl`; on Linux install it once:
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

### ✅ Check (Windows and Linux)
```bash
kubectl config use-context docker-desktop
kubectl get nodes
```
You should see one node with STATUS **Ready**, called **docker-desktop** (Kubeadm cluster) or
**docker-desktop-control-plane** (kind cluster). Note which one: kind needs one extra step in Part 4.

`error: current-context is not set` / `no context exists with the name "docker-desktop"`? Your `~/.kube/config`
is empty. Docker Desktop → **Kubernetes** → **Reset cluster** (or restart Docker Desktop), then run the check again.

---

## Part 4 — Run the app on Kubernetes

From the course folder:

```bash
cd Day4_Docker_Kubernetes_Cloud
```

**Step 0 — kind cluster only** (node named `docker-desktop-control-plane`; skip on Kubeadm).
A kind node runs in its own container with its own image store, so it can't see the images from Part 2
(pods would stay in `ImagePullBackOff`). Copy them in once (≈ 30 s; again after every rebuild):

```powershell
# 🪟 PowerShell
foreach ($n in (docker ps --filter "label=io.x-k8s.kind.cluster" --format "{{.Names}}")) { foreach ($i in "cifar-api:v1","cifar-ui:v1") { docker save $i -o img.tar; docker cp img.tar "${n}:/img.tar"; docker exec $n ctr -n k8s.io images import /img.tar; docker exec $n rm /img.tar } }; Remove-Item img.tar
```
```bash
# 🐧 Linux / Git Bash
for n in $(docker ps --filter "label=io.x-k8s.kind.cluster" --format "{{.Names}}"); do for i in cifar-api:v1 cifar-ui:v1; do docker save $i | MSYS_NO_PATHCONV=1 docker exec -i $n ctr -n k8s.io images import -; done; done
```
Check: `docker exec docker-desktop-control-plane crictl images` lists `docker.io/library/cifar-api v1` and `cifar-ui v1`.

Then deploy:

```bash
kubectl apply -k k8s/
kubectl -n ml-demo get pods -w
```

Watch the pods go from `ContainerCreating` to `Running` with READY **1/1** (about 30–60 s).
Then press **Ctrl + C** to stop watching.

### ✅ Check
Open **http://localhost:30080** → pick a picture → prediction. 🎉
Under the prediction you see **"Served by pod: cifar-api-…"** — press *Predict* again a few times: the name
changes, because Kubernetes spreads the requests over the **2 copies** of the API.

**Try self-healing (30 s):**
```bash
kubectl -n ml-demo get pods
kubectl -n ml-demo delete pod <copy-one-cifar-api-name-here>
kubectl -n ml-demo get pods
```
A new API pod appears immediately — Kubernetes keeps 2 copies running, whatever happens.

### Remove the app from Kubernetes (when you are done)
```bash
kubectl delete -k k8s/
```
To switch Kubernetes off completely: Docker Desktop → **Kubernetes** → **Stop**.

---

## If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `'docker' is not recognized` / `command not found` | Docker Desktop not installed, or terminal opened before installing | install (Part 1), open a **new** terminal |
| `failed to connect to the docker API` / `Cannot connect to the Docker daemon` | Docker Desktop is not running | open it, wait for **"Engine running"** |
| Linux: Docker Desktop won't start, mentions **KVM** | virtualization off, or you did not log out/in after `usermod` | turn on VT-x/AMD-V in BIOS; log out and in |
| `port is already allocated` / `address already in use` | another program uses port 5000, 8501 or 30080 — often the **Day 3 API/Streamlit** still running | stop it (Ctrl + C in its terminal) |
| `no configuration file provided` / `must build kustomization` | terminal is in the wrong folder | Part 2 runs in `Day4_Docker_Kubernetes_Cloud/ml-app`, Part 4 in `Day4_Docker_Kubernetes_Cloud` |
| `error validating "k8s/": ... failed to download openapi ... EOF` or `connection refused` | kubectl talks to a cluster that is **not running** (or an old one, e.g. minikube) | `kubectl config use-context docker-desktop`, check Kubernetes is **running** in Docker Desktop |
| Pods stuck in `ErrImagePull` / `ImagePullBackOff` | Kubernetes can't find `cifar-api:v1` | do Part 2 first (it builds the images); on a **kind** cluster do Part 4 step 0 |
| `error: current-context is not set` | `~/.kube/config` is empty | Docker Desktop → Kubernetes → **Reset cluster** (or restart Docker Desktop) |
| `invalid JSON patch` (PowerShell) | PowerShell 5.1 breaks `"{\"…\"}"` quoting | use the YAML form: `-p 'data: {DEFAULT_MODEL: fp32}'` |
| Pods `Running` but READY `0/1` for a long time | the API is still loading the model | wait 1 minute; `kubectl -n ml-demo describe pod <name>` shows why |
| Page says **"API not reachable"** | the API is still starting | wait 30 s, refresh |
