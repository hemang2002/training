# 01 · Push images to Docker Hub

Most cloud options (Render, Killercoda, any Kubernetes cluster that isn't on your laptop) **pull** images from
a registry. Docker Hub's free plan allows public repositories — perfect for a demo.

> ⚠️ Public images are visible to anyone. Our image contains only code + a CIFAR-10 model: fine.
> Never push images containing secrets, API keys, or private data.

## Steps

1. Create a free account at <https://hub.docker.com/signup>.
2. Log in from the terminal (use a **Personal Access Token** as password: Account settings → Personal access tokens):
   ```bash
   docker login -u <your-user>
   ```
3. Tag and push (from `Day4_Docker_Kubernetes_Cloud/ml-app/`):
   ```bash
   docker tag cifar-api:v1 <your-user>/cifar-api:v1
   docker tag cifar-ui:v1  <your-user>/cifar-ui:v1
   docker push <your-user>/cifar-api:v1
   docker push <your-user>/cifar-ui:v1

   # optional: all-in-one image for Render
   docker build -f Dockerfile.allinone -t <your-user>/cifar-allinone:v1 .
   docker push <your-user>/cifar-allinone:v1
   ```
4. Check them at `https://hub.docker.com/r/<your-user>/cifar-api`.

## CPU architecture: amd64 vs arm64

Images built on an Intel/AMD laptop are `linux/amd64`. They will **not** run on ARM machines
(Oracle Ampere A1 VMs, Raspberry Pi, AWS Graviton) — error: `exec format error`.
Build a multi-architecture image with buildx (Docker Desktop includes it):

```bash
docker buildx create --use --name multi       # once
docker buildx build --platform linux/amd64,linux/arm64 \
  -f api/Dockerfile -t <your-user>/cifar-api:v1 --push .
docker buildx build --platform linux/amd64,linux/arm64 \
  -f ui/Dockerfile -t <your-user>/cifar-ui:v1 --push .
```

(Or simply build the images **on** the ARM VM itself — see `04_vm_docker_compose.md`.)

## Use them in Kubernetes

Uncomment the `images:` block at the bottom of `k8s/kustomization.yaml`, put your user name in, then
`kubectl apply -k k8s/`. Kustomize rewrites `cifar-api:v1` → `docker.io/<your-user>/cifar-api:v1` everywhere.
