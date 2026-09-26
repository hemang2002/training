# 05 · Real Kubernetes on a free VM with k3s

**k3s** is a lightweight, CNCF-certified Kubernetes distribution (single binary, runs in ~512 MB+).
On a free VM it gives you a real cluster where **our exact manifests from `k8s/` run unchanged**.

Recommended VM: Oracle Cloud Always Free **Ampere A1** (ARM, generous RAM) — see `cloud/README.md`.
A 1 GB VM (e2-micro / E2.1.Micro) is too tight for k3s + our app; use it for `04_vm_docker_compose.md` instead.

> 💸 Stay on free-eligible shapes, set a budget alert, delete resources after the demo.

## 1. VM + firewall

Create an Ubuntu VM as in `04_vm_docker_compose.md` §1 and open inbound TCP **30080** (our UI NodePort)
in the provider firewall (+ the iptables rule on Oracle, with port 30080).

## 2. Install k3s (one command)

```bash
ssh ubuntu@<public-ip>
curl -sfL https://get.k3s.io | sh -
sudo k3s kubectl get nodes          # STATUS Ready after ~30 s

# use plain `kubectl` without sudo
mkdir -p ~/.kube && sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config && sudo chown $USER ~/.kube/config
export KUBECONFIG=~/.kube/config && echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc
kubectl get pods -A                 # coredns, traefik, metrics-server, local-path-provisioner
```

k3s already includes **metrics-server** (HPA works) and the **Traefik** ingress controller.

## 3. Get the images into k3s

k3s uses **containerd**, not Docker — it can't see images built with `docker build` unless you import them.

**Option A — build on the VM and import (no registry needed):**
```bash
curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER && newgrp docker
# copy ml-app/ and k8s/ from your laptop:  scp -r ml-app k8s ubuntu@<public-ip>:~/
cd ~/ml-app
docker build -f api/Dockerfile -t cifar-api:v1 .
docker build -f ui/Dockerfile  -t cifar-ui:v1 .
docker save cifar-api:v1 | sudo k3s ctr images import -
docker save cifar-ui:v1  | sudo k3s ctr images import -
sudo k3s ctr images ls | grep cifar
```

**Option B — pull from Docker Hub:** push multi-arch images (`01_docker_hub.md`), then uncomment the
`images:` block in `k8s/kustomization.yaml`.

## 4. Deploy — the same YAML as on minikube

```bash
cd ~
kubectl apply -k k8s/
kubectl -n ml-demo get pods -w
kubectl -n ml-demo get svc cifar-ui        # NodePort 30080
```

Open `http://<public-ip>:30080`. Everything from README §5 (self-healing, scaling, rollout, HPA + load test Job) works here too:

```bash
kubectl -n ml-demo create configmap loadtest-script --from-file=k8s/loadtest/loadtest.py
kubectl apply -f k8s/loadtest/loadtest-job.yaml
kubectl -n ml-demo get hpa -w
```

## 5. Optional: Ingress on port 80 with Traefik

Edit `k8s/extras/ingress.yaml`: set `ingressClassName: traefik` and replace `host: cifar.local` with
`<public-ip>.nip.io` (nip.io is a free wildcard DNS that maps the name to that IP). Apply it, open port 80
in the firewall, and browse to `http://<public-ip>.nip.io`.

## 6. Manage the cluster from your laptop (optional)

```bash
# on the VM: print the kubeconfig
sudo cat /etc/rancher/k3s/k3s.yaml
```
Copy it to your laptop as `~/.kube/k3s-demo.yaml`, replace `127.0.0.1` with the VM's public IP, open TCP 6443
in the firewall **for your IP only**, then `kubectl --kubeconfig ~/.kube/k3s-demo.yaml get nodes`.
(For a classroom demo, SSH-ing into the VM is simpler and safer.)

## 7. Clean up

```bash
kubectl delete -k k8s/
/usr/local/bin/k3s-uninstall.sh      # removes k3s completely
```
Then terminate the VM.

## Talking points

* Same manifests, different cluster: that's the portability promise of Kubernetes.
* Managed Kubernetes (GKE/AKS/EKS/OKE) = someone else runs the control plane; you still write the same YAML.
  `type: LoadBalancer` services on managed clusters create a paid cloud load balancer — that's why we use NodePort here.
* A single-node cluster still has one point of failure: the VM.
