# 06 · Killercoda — Kubernetes in the browser (no install, no cloud account)

**Why:** students without admin rights / weak laptops get a real kubeadm Kubernetes cluster in a browser tab,
free, in seconds. Great backup plan if minikube fails on someone's machine.

**Trade-offs:** sessions are temporary (they expire, everything is wiped), limited resources, you must sign in,
and the cluster can't see your laptop's images → images must come from a public registry (Docker Hub).
Current session limits: see `cloud/README.md`.

## Steps

1. Push the images to Docker Hub (`01_docker_hub.md`) — `linux/amd64` is fine here.
2. Open <https://killercoda.com/playgrounds> → sign in → choose the **Kubernetes** playground
   (a control-plane node + a worker node).
3. Get the manifests into the session. Easiest: clone your repo
   ```bash
   git clone https://github.com/<you>/<course-repo>.git && cd <course-repo>/Day4_Docker_Kubernetes_Cloud
   ```
   (or paste the YAML files with `cat > file.yaml <<'EOF' … EOF`).
4. Point the manifests at Docker Hub — edit `k8s/kustomization.yaml` and uncomment the `images:` block with your
   user name. Then:
   ```bash
   kubectl apply -k k8s/
   kubectl -n ml-demo get pods -w
   ```
5. Metrics server (for the HPA) may not be installed in the playground:
   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
   kubectl -n kube-system patch deployment metrics-server --type=json \
     -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
   ```
6. Open the UI: Killercoda's menu (top right) → **Traffic / Ports** → enter `30080` → it opens the NodePort
   in a new tab through Killercoda's proxy.

Everything in README §5.3–5.6 (self-healing, scaling, rollout/rollback, HPA with the load test Job) can be done here.
