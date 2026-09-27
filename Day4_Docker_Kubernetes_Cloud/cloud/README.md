# Free cloud options for the Day 4 demo

> Goal: show **how the app looks when deployed** — a public URL, logs, a cluster — not a production website.
> Everything below is free-tier or free-credit only. Offers change often: facts verified **2026-09-26**;
> re-check the official page the week before class. Full comparison with sources:
> [`References/free_cloud_options.md`](../../References/free_cloud_options.md).

## Recommendation

| Goal | Use | Card needed? | Guide |
|---|---|---|---|
| ⭐ Public URL for the Dockerized app, quickest | **Render** free web service (all-in-one image) | No (payment method optional) | [03_render.md](03_render.md) |
| ⭐ Real Kubernetes in class, no cloud account | **minikube / kind** locally (README §5); **Killercoda** as zero-install fallback | No | [06_killercoda_playground.md](06_killercoda_playground.md) |
| "Real server": SSH + docker compose on a VM | Oracle Always Free / Google Cloud free tier / Azure for Students | Oracle, GCP: yes · Azure for Students: no | [04_vm_docker_compose.md](04_vm_docker_compose.md) |
| Real Kubernetes in the cloud with our manifests | k3s on Oracle Always Free Arm VM (or Azure for Students VM) | Oracle: yes · Azure Students: no | [05_vm_k3s_kubernetes.md](05_vm_k3s_kubernetes.md) |
| You already have Hugging Face PRO | Hugging Face Docker Space | — (PRO plan) | [02_huggingface_spaces.md](02_huggingface_spaces.md) |

**Registry for all remote options:** Docker Hub free public repos — [01_docker_hub.md](01_docker_hub.md).

## Key facts (verified 2026-09-26)

| Option | What's free | Card | Docker | Kubernetes | Gotchas |
|---|---|---|---|---|---|
| **Render** | Free web service: 512 MB RAM, 0.1 CPU, 750 instance-hours/month per workspace | Not required to start | ✅ Dockerfile or registry image | ❌ | Spins down after **15 min** idle, ~**1 min** cold start; local files lost on restart; 0.1 CPU → inference is slower than on your laptop |
| **Hugging Face Spaces** | CPU Basic hardware (2 vCPU, 16 GB) has no hourly cost | — | ⚠️ **Creating Docker (and Gradio) Spaces now requires a paid plan (PRO)**; static Spaces are free | ❌ | Sleeps when unused; free accounts get static Spaces + 2 ZeroGPU Gradio Spaces only |
| **Oracle Cloud Always Free** | Arm A1 VMs **2 OCPU / 12 GB** total (halved in June 2026 — older blogs say 4/24), 200 GB block storage, basic OKE cluster (no control-plane fee) | Yes (verification) | ✅ | ✅ k3s or OKE | Arm CPU → build images on the VM or multi-arch; idle VMs can be reclaimed; open both VCN security list **and** OS iptables |
| **Google Cloud** | $300 / 90-day trial; always-free e2-micro VM (1 GB, US regions); GKE free tier covers one cluster's management fee, **not** nodes | Yes | ✅ | ✅ (nodes paid from credit) | e2-micro is too small for k3s — use compose + swap |
| **Azure for Students** | $100 credit / 12 months (renewable while a student), school email | **No** | ✅ | ✅ AKS free control plane, nodes from credit | Delete resources after class to save credit |
| **AWS (new accounts)** | Credit-based Free plan ($100 + up to $100 earned, max 6 months) | Yes | ✅ | EKS is paid — verify before class | The old "12 months t2.micro" model no longer applies to new accounts |
| **Killercoda** | Browser Linux playground (Ubuntu), ~1 h sessions | No (sign-in) | ✅ build images there | ✅ k3s on that Docker (temporary) | No Docker Hub needed; repo must be public on GitHub; everything wiped at session end |
| Play with Docker / Play with Kubernetes | — | — | — | — | **Shut down on 1 March 2026** — don't use old tutorials that point to them |

## Budget safety checklist (for any cloud with a card)

1. Create resources only from the *Always Free* / *Free tier* labelled options.
2. Set a **budget alert** (e.g. $1) in the billing console before creating anything.
3. Avoid `type: LoadBalancer` Services and managed databases — those cost money.
4. **Delete** VMs, disks, public IPs and clusters right after the demo.
