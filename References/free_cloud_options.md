# Free Cloud Options for Demoing a Dockerized Flask + Streamlit ML App and a Small Kubernetes Cluster

**Verified on: 2026-09-26** against the official pages linked below, using live web fetches on that date.
**Scope:** classroom **demos**, not production. Free tiers change often, sometimes without notice (Oracle, Hugging Face, and DigitalOcean all changed in 2026). **Re-check the official link the week before class.**
Anything marked **"verify before class"** could not be confirmed on an official page, or official and third-party sources disagreed.

---

## 1. Summary table

| Option | What's free (key limits) | Credit card? | Docker? | Kubernetes? | Verdict for this course |
|---|---|---|---|---|---|
| **Render (free web service)** | 512 MB RAM / 0.1 CPU; spins down after 15 min idle (~1 min cold start); 750 free instance-hours per workspace per month; ephemeral disk | Not required to start, per Render's own marketing page (verify before class); if you go over the limits without a card, Render suspends your free services | **Yes** (builds from a Dockerfile or pulls a prebuilt image) | No | **Best no-card Docker demo** |
| **Hugging Face Spaces (Docker SDK)** | CPU Basic hardware: 2 vCPU / 16 GB RAM / 50 GB non-persistent disk; sleeps when unused. **But creating Docker/Gradio Spaces now needs a paid plan (PRO, $9/mo)** | No card for a free account, but a free account **can't create Docker Spaces** | Yes (PRO) | No | **No longer free for Docker.** Fine if the trainer pays for PRO |
| **Oracle Cloud Always Free** | Ampere A1 Arm: **2 OCPU + 12 GB RAM total** (1,500 OCPU-h + 9,000 GB-h per month), halved from 4/24 in June 2026; 2 AMD micro VMs (1 GB each); 200 GB block storage; 1 flexible LB (10 Mbps) + 1 NLB; $300 trial for 30 days | **Yes** (verification hold) | Yes (on a VM) | **Yes**: OKE *basic* cluster has no control-plane fee; A1 worker nodes fit in Always Free | **Best "real" managed K8s at $0**, if you have a card and can get A1 capacity |
| **Google Cloud** | $300 / 90-day trial; Free Tier: 1 e2-micro VM (us-west1/us-central1/us-east1), 30 GB disk, 1 GB egress; Cloud Run 2M requests/month; **GKE: $74.40/month credit = 1 zonal or Autopilot cluster fee** (nodes/compute not covered) | **Yes** (trial and Free Tier) | Yes (Cloud Run, VM) | Yes (GKE, with nodes paid from trial credit) | Great with the $300 trial. **Faculty can request education credits (no card)** |
| **AWS Free Tier (new accounts since 15 Jul 2025)** | $100 credit at signup + up to $100 more earned by using services (max $200); Free plan ends at **6 months or when credits run out**; 30+ always-free services; data kept 90 days after expiry | **Yes** (identity verification, per AWS builder/third-party sources; verify before class) | Yes (EC2, ECS, App Runner, Lightsail) | EKS: paid control plane; the Free plan restricts some services (**verify before class**) | Usable, but credit-based and time-boxed |
| **Azure for Students** | **$100 credit, 12 months**, renewable yearly while a student; free services | **No card** | Yes (App Service, Container Apps, ACI) | **Yes**: AKS Free tier has free cluster management, and nodes come out of the $100 | **Best for students with a school email** |
| **Azure free account** | $200 credit for 30 days + 12 months of popular free services + 65+ always-free services | **Yes** (plus phone) | Yes | Yes (AKS Free tier; nodes billed against credit) | OK for the trainer; students should use Azure for Students |
| **GitHub Student Developer Pack** | Azure $100 (18+); Heroku $13/month for 24 months; Codespaces Pro; **DigitalOcean left the pack, and all its credits expired on 1 Aug 2026** | Depends on the offer | Via partners | Via Azure | Good add-on for students |
| **Killercoda** | Free browser-based K8s playgrounds (kubeadm cluster); **1-hour session limit** on free (PLUS: up to 4 h); captcha challenges | No | Yes (inside the VM) | **Yes** (real cluster, ephemeral) | **Best zero-install K8s lab** |
| **Play with Kubernetes / Play with Docker** | **Discontinued: "unavailable starting March 1, 2026"** (notice on both sites) | n/a | n/a | n/a | **Do not use** |
| **Koyeb** | 1 free instance per org: 512 MB / 0.1 vCPU / 2 GB SSD, Frankfurt or Washington D.C., scales to zero after 1 h idle | **Yes**, with a **$29 pre-authorisation hold** (official FAQ) | Yes | No | Not recommended: card required. Third-party reports say new signups start on Pro after the Feb 2026 Mistral AI acquisition (verify before class) |
| **Railway** | Trial: one-time $5 grant for up to 30 days (1 GB RAM, shared vCPU, 5 services/project), then Free plan with $1 credit/month | Not stated on the trial page (verify before class) | Yes | No | Too little credit for a class |
| **Fly.io** | **No free tier.** Free trial = 2 VM-hours **or** 7 days, whichever comes first; trial machines auto-stop after 5 min | Card required for organisations; adding a card ends the trial | Yes | No | Not suitable |
| **Google Colab (training only)** | Free notebooks run at most 12 h; GPUs "heavily restricted" and not guaranteed; idle timeouts | No | No | No | **Use for training** the models, not for serving |
| **Streamlit Community Cloud (bonus)** | Free hosting of Streamlit apps from a GitHub repo; apps sleep after 12 h without traffic (Streamlit docs, verify before class) | No | **No** (not Docker) | No | Easy no-card fallback for the Streamlit UI only |
| **Local: minikube / kind** | Free on students' laptops (needs Docker Desktop or an equivalent) | No | Yes | **Yes** | **Primary K8s teaching environment** |

---

## 2. Recommendations

| Need | Recommendation | Why | Fallback |
|---|---|---|---|
| **Best for a Docker demo without a credit card** | **Render free web service** (deploy from Dockerfile) | Official Docker support, no card needed to start, HTTPS URL, auto-deploy from GitHub | Streamlit Community Cloud (not Docker); HF Spaces only if the trainer buys PRO |
| **Best for real Kubernetes, free** | **In class: kind or minikube locally, plus Killercoda for anyone who can't install.** **In the cloud: Oracle OKE basic cluster on Always Free A1 nodes** | Local is always available and has no limits. OKE basic has no control-plane fee, and A1 (2 OCPU / 12 GB) can run 1–2 small worker nodes at $0 | GKE (free cluster fee + $300 trial for nodes), or AKS Free tier on Azure for Students credit |
| **Best for students with a .edu / school email** | **Azure for Students** ($100, 12 months, **no card**) | No card is the biggest barrier removed. Covers App Service/Container Apps for Docker and AKS (Free tier) for K8s | GitHub Student Developer Pack (includes the Azure offer); **trainer applies for Google Cloud education credits** (faculty: up to $100 per teaching staff and $50 per student, no card for students) |
| **Best for training models** | **Google Colab** (free) | GPU available intermittently; notebooks up to 12 h | Kaggle notebooks (not verified here) |

---

## 3. Details per option

### 3.1 Hugging Face Spaces (Docker SDK). **Changed in 2026**
- **Official:** https://huggingface.co/docs/hub/spaces-overview · https://huggingface.co/docs/hub/spaces-sdks-docker · https://huggingface.co/pricing
- **What's free:** Static Spaces are free for everyone. **"Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts, Team or Enterprise for organizations."** Free personal accounts in good standing (verified email, account older than 30 days) can host up to **2 ZeroGPU Gradio Spaces**, with a 5-minute daily GPU quota.
- A forum thread dated **9 Jul 2026** reports new free accounts see Docker marked **"Paid"** and cannot select CPU Basic: https://discuss.huggingface.co/t/new-free-accounts-cannot-create-cpu-basic-gradio-spaces-only-zerogpu-available/177629
- **Hardware (CPU Basic, no hourly cost):** 2 vCPU, 16 GB RAM, 50 GB disk (**not persistent**; lost on restart unless you attach a Storage Bucket).
- **Sleep:** "On free hardware, your Space will 'go to sleep'… after a period of time if unused." The docs don't state the exact duration (verify before class).
- **PRO price:** $9/month.
- **Docker specifics:**
  - Put `sdk: docker` in the YAML header of `README.md`. Default port is **7860**; change it with `app_port`.
  - The container runs as **UID 1000**, so create a user, `COPY --chown=user`.
  - Only **one externally exposed port**. To serve Flask and Streamlit together you need an Nginx reverse proxy, or have Streamlit call Flask on localhost.
  - Outbound network requests are allowed only on ports **80, 443, 8080**.
  - Secrets are available as env vars at runtime. At build time, use `RUN --mount=type=secret`.
- **Gotchas:** no GPU during `docker build`; don't do recursive `chown` (it bloats the image).
- **Recommendation:** still the nicest ML demo UX. Use it if the trainer pays for PRO; don't make it a student requirement.

### 3.2 Render: free web services
- **Official:** https://render.com/docs/free · https://render.com/docs/docker · https://render.com/docs/deploy-flask
- **What's free:**
  - Free instance: **512 MB RAM, 0.1 CPU**. The spec is from Render's search results and community posts; the /docs/free page itself omits RAM/CPU (verify before class).
  - Free for web services, static sites, Postgres and Key Value.
  - Deploy with **Docker**, Node, Python, Ruby, Go, Rust or Elixir.
- **Sleep:** spins down after **15 minutes without inbound traffic** (HTTP or WebSocket). Spin-up takes **about one minute**.
- **Hours:** **750 free instance-hours per workspace per calendar month**. If exhausted, all free web services are suspended until the next month.
- **Credit card:** Render's page "Platforms with a real free tier… 2026" says no credit card is required (verify before class). If you exceed included usage *without* a payment method, Render suspends your free services rather than billing you.
- **Gotchas:**
  - The filesystem is **ephemeral**: lost on redeploy, restart and spin-down.
  - Outbound SMTP ports 25/465/587 are blocked. No one-off jobs.
  - Free Postgres **expires 30 days** after creation, then a 14-day grace period.
  - Build minutes and bandwidth count against monthly allowances.
  - A PyTorch image can exceed 512 MB RAM at runtime. **Use ONNX Runtime + a quantized model, or CPU-only torch wheels**, to fit.
- **Flask + Streamlit:** deploy as **two separate web services**. They share the 750 h/month pool, which is fine because spin-down stops the clock. Point Streamlit at Flask's public `https://…onrender.com` URL through an env var.

### 3.3 Oracle Cloud Infrastructure: Always Free (+ OKE)
- **Official:**
  - https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm
  - https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm
  - https://www.oracle.com/cloud/free/faq/
  - OKE basic vs enhanced: https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengcomparingenhancedwithbasicclusters_topic.htm
- **What's free (per the docs on 2026-09-26):**
  - **Ampere A1 (Arm):** 1,500 OCPU-hours + 9,000 GB-hours per month, "equivalent to **2 OCPUs and 12 GB of memory**" for Always Free tenancies.
  - **AMD:** up to 2 × VM.Standard.E2.1.Micro (1/8 OCPU, 1 GB RAM each).
  - **Storage:** 200 GB total block + boot storage, 5 backups, 20 GB Object Storage.
  - **Load balancing:** 1 Flexible Load Balancer (10 Mbps) + 1 Network Load Balancer.
- **2026 change:** the A1 allowance was **halved from 4 OCPU / 24 GB to 2 OCPU / 12 GB, effective about 15 Jun 2026**, with no announcement (InfoQ, Jul 2026: https://www.infoq.com/news/2026/07/oracle-cloud-free-tier-limits/). Older blog posts and tutorials still say 4/24.
- **Trial:** US$300 credit, usable for **30 days**, in select countries. After that you keep Always Free.
- **Credit card:** **required for identity verification.** There's a temporary hold, and you aren't charged unless you upgrade.
- **Kubernetes (OKE):**
  - **Basic clusters have no control-plane fee.** Enhanced clusters cost $0.10/cluster/hour, up to $74.40/month.
  - Use the **VM.Standard.A1.Flex** shape for worker nodes to stay within Always Free, e.g. 2 nodes × 1 OCPU / 6 GB.
  - Load-balancer Services can use the Always Free LB.
- **Gotchas:**
  - **"Out of host capacity"** errors for A1 in popular home regions are common. Pick your home region carefully; it can't be changed.
  - **Idle reclamation:** instances are reclaimed if, over 7 days, 95th-percentile CPU, network *and* memory (A1) all stay below 20%.
  - Container images must be **arm64** (build multi-arch with `docker buildx`).
  - Many users report having to upgrade to Pay-As-You-Go (still $0 within Always Free) before OKE or A1 provisioning works reliably (verify before class).

### 3.4 Google Cloud: Free Tier, $300 trial, GKE free tier
- **Official:** https://docs.cloud.google.com/free/docs/free-cloud-features · https://cloud.google.com/kubernetes-engine/pricing · Education: https://cloud.google.com/edu/students , https://cloud.google.com/edu/faculty
- **$300 Free Trial:** a **90-day** program. You **must provide a credit card or other payment method**; a $0–$1 temporary authorisation hold is placed.
- **Compute Engine Free Tier:** 1 non-preemptible **e2-micro** VM per month in **us-west1, us-central1 or us-east1**; 30 GB-month standard persistent disk; 1 GB/month egress from North America.
- **Cloud Run Free Tier:** 2M requests/month, 360,000 GB-seconds of memory, 180,000 vCPU-seconds, 1 GB egress from North America. Very good for a **serverless Docker demo** (scales to zero), but it needs a billing account.
- **GKE:** "One free Autopilot or zonal Standard cluster per month," i.e. **$74.40/month in credits per billing account**. It "applies to the cluster charge only," so **nodes, networking and load balancers are billed**; cover them with trial credit.
- **Education (no card for students):**
  - Eligible **faculty can apply for up to $100 in Cloud credits per teaching staff and up to $50 per student**, plus Google Skills lab credits.
  - Students verify with their school email and receive a coupon code.
  - Application approval and country eligibility vary (verify before class; apply weeks in advance).
- **Gotchas:** e2-micro (1 GB RAM) is too small for PyTorch plus two apps; use ONNX Runtime. Set **budget alerts**. Delete GKE load balancers after class.

### 3.5 AWS Free Tier: credit-based model for accounts created on or after 15 Jul 2025
- **Official:** https://aws.amazon.com/free/ · FAQ: https://aws.amazon.com/free/free-tier-faqs/ · Announcement: https://aws.amazon.com/about-aws/whats-new/2025/07/aws-free-tier-credits-month-free-plan/
- **What's free:**
  - **$100 in credits at signup**, plus up to **$100 more** earned by trying key services, for a total of **up to $200**.
  - Choose a **Free plan** (no charges unless you upgrade) or a **Paid plan** (all services, billing beyond credits).
  - The Free plan account "closes on its own **6 months** after you open it or when your credits run out, whichever comes first."
  - AWS retains data **90 days** after the free plan expires; upgrade within that window to restore it.
  - 30+ always-free services within monthly limits.
- **Free plan restriction:** it is "limited from accessing a subset of AWS services… that would immediately consume the entire Free Tier credit amount or require hardware purchases." Whether **EKS** is available on the Free plan wasn't confirmed (verify before class).
- **Credit card:** AWS builder-center and third-party sources say a valid card is required for identity verification even on the Free plan. The official FAQ page didn't state it explicitly (verify before class).
- **Docker:** EC2, ECS/Fargate, App Runner and Lightsail containers all work (billed against credits).
- **Kubernetes:** EKS control plane is billed hourly. EKS pricing wasn't re-verified here (verify before class). It's better to run k3s/kind on a single EC2 instance, paid from credits.
- **Gotchas:** old tutorials describe the pre-2025 "12 months free t2.micro" model, which **doesn't apply to new accounts**.

### 3.6 Microsoft Azure: Azure for Students, Azure free account, AKS Free tier
- **Official:** https://azure.microsoft.com/en-us/free/students/ · https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account · https://azure.microsoft.com/en-us/pricing/free-services · https://learn.microsoft.com/en-us/azure/aks/free-standard-pricing-tiers
- **Azure for Students:**
  - "**$100 credit** to use on Azure services within **12 months**." **"No credit card required."**
  - "Available only to **full-time university students**." Sign up with your school email.
  - "Renew your subscription annually… as long as you're a student."
  - There's also a 13–17 offer via GitHub (limited services).
- **Azure free account (non-students):** **$200 credit, 30 days**; 12 months of free popular services; 65+ always-free services; new customers only. **Needs a phone plus a credit/debit card** (non-prepaid). You may see a $1 verification hold.
- **AKS Free tier:**
  - "Free cluster management. Pay as you go for consumed resources." No financially backed SLA.
  - Recommended for **fewer than 10 nodes**. Create with `az aks create --tier free`.
  - **Nodes (VMs) are billed.** With the student $100, run 1 small node and stop the cluster (`az aks stop`) after class.
- **Docker options:** App Service (containers), Azure Container Apps, Azure Container Instances. Free grants weren't re-verified (verify before class).
- **Gotchas:** some universities' emails aren't auto-recognised, so have students test signup before the deployment session. AKS "Automatic" SKU uses the Standard (paid) tier by default, so choose **Base + Free**.

### 3.7 GitHub Student Developer Pack
- **Official:** https://education.github.com/pack
- **Relevant offers (verified 2026-09-26):**
  - **Microsoft Azure**: "Free access to 25+ Microsoft Azure cloud services plus $100 in Azure credit" (18+).
  - **Heroku**: "$13 USD per month for 24 months."
  - **GitHub Codespaces**: free Pro-level access. A cloud dev box that students can use to run Docker/kind if their laptops are weak (verify Docker-in-Codespaces resources before class).
  - LocalStack (AWS emulator) and New Relic.
- **DigitalOcean has left the pack.** Every DO pack credit, new or already redeemed, **expired 1 Aug 2026**: https://github.com/orgs/community/discussions/201240 . Ignore older guides recommending "$200 DigitalOcean for students".

### 3.8 Killercoda (browser Kubernetes playground)
- **Official:** https://killercoda.com/playgrounds · Kubernetes playground: https://killercoda.com/playgrounds/scenario/kubernetes · Pricing: https://killercoda.com/pricing
- **What's free:** all free scenarios and playgrounds, unlimited use. The Kubernetes playground is a kubeadm cluster, typically `controlplane` + `node01` (verify the current layout before class).
- **Limit:** free sessions last **1 hour**. PLUS gives "up to 4 hours" and no captcha. The 1-hour figure comes from third-party sources plus the PLUS comparison (verify before class).
- **Credit card:** no. Login needed.
- **Great for:** Deployments, Services, probes, HPA (install metrics-server) when laptops can't run minikube. **Everything is lost at session end.** Keep manifests in a GitHub repo and `kubectl apply -f <raw URL>`.

### 3.9 Play with Kubernetes / Play with Docker: **DISCONTINUED**
- https://labs.play-with-k8s.com/ and https://labs.play-with-docker.com/ both show: **"… will be unavailable starting March 1, 2026. Visit Docker Docs for supported labs and guides."**
- Remove them from slides and handouts. Use **Killercoda** instead.

### 3.10 Koyeb
- **Official:** https://www.koyeb.com/docs/reference/instances · https://www.koyeb.com/docs/faqs/pricing · https://www.koyeb.com/pricing
- **Free instance:** 512 MB RAM, 0.1 vCPU, 2 GB SSD. One per organisation, in Frankfurt or Washington D.C. Scales to zero after **1 hour** without traffic. No volumes, no workers.
- **Credit card: required.** Official FAQ: "We require a credit card to prevent fraud and abuse"; a **$29 pre-authorisation hold**, then a prorated charge for the selected plan (typically Pro at signup). You can downgrade to Starter (pay-as-you-go, which includes the free instance).
- **2026 change:** Koyeb was acquired by **Mistral AI (Feb 2026)**. Third-party reviews say new accounts now start on Pro (~$29/mo) (verify before class). **Not recommended for students.**

### 3.11 Railway
- **Official:** https://docs.railway.com/pricing/free-trial
- **Trial:** "access to basic features for up to 30 days and includes a one-time grant of **$5**." Limited to 1 GB RAM, shared vCPU, 5 services per project.
- **After the trial:** Free plan with **$1 of free credit per month**.
- **Credit card:** not stated on the page (verify before class). `ssh railway.new` lets you try without an account.
- **Verdict:** $1/month won't keep two containers running. Skip it.

### 3.12 Fly.io
- **Official:** https://docs.fly.io/about/free-trial/ · https://docs.fly.io/about/pricing/
- **No free tier for new users.** The trial is "**2 hours of machine runtime or 7 days of access**, whichever comes first." Up to 10 machines, 2 vCPU / 4 GB each, and trial machines auto-stop after 5 minutes.
- "All organizations (except for Linked Organizations) require a credit card on file." **Adding a card ends the free trial** and starts billing.
- **Verdict:** not suitable for a class.

### 3.13 Google Colab (for the training phase)
- **Official FAQ:** https://research.google.com/colaboratory/faq.html
- Free notebooks run "for **at most 12 hours**, depending on availability and your usage patterns."
- GPU access is "heavily restricted" and GPU types vary. Idle sessions are terminated, and limits "fluctuate."
- **Use:** train or fine-tune (transfer learning, Faster R-CNN, DCGAN), export to `.pt`/`.onnx`, download, then serve elsewhere. Save checkpoints to Google Drive often.

### 3.14 Streamlit Community Cloud (bonus, no card)
- **Official:** https://docs.streamlit.io/deploy/streamlit-community-cloud
- Free hosting of Streamlit apps straight from GitHub. **No Docker, no custom backend container.** Apps sleep after 12 h without traffic (verify before class).
- **Use:** host the Streamlit front end with no card, calling a Flask API hosted on Render.

---

## 4. Practical tips for the Day 4 deployment session

1. **Shrink the model first.** An ONNX + INT8 model served by `onnxruntime` fits free-tier RAM (Render 512 MB, e2-micro 1 GB). A full `torch` install often won't. Use CPU-only wheels: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
2. **One image, two processes, or two images?**
   - On Render or Azure: two services (Flask API + Streamlit UI), wired together with an env var `API_URL`.
   - On HF Spaces (single port): one container with Nginx, or Streamlit calling Flask on `localhost`.
3. **Arm vs x86:** Oracle A1 is **arm64**. Build with `docker buildx build --platform linux/amd64,linux/arm64`.
4. **Health endpoints:** add `/healthz` to Flask. Render health checks and K8s liveness/readiness probes both use it.
5. **Cost hygiene:** set budget alerts (GCP/AWS/Azure). `az aks stop` / delete the GKE cluster and its load balancers after class. Oracle reclaims idle VMs.
6. **Plan B for K8s:** if cloud signup fails in class, fall back to **kind/minikube locally → Killercoda**.

---

## 5. Sources checked on 2026-09-26
Official pages: all links in section 3.
Third-party sources, used only where flagged "verify before class":
- InfoQ on the Oracle A1 cut: https://www.infoq.com/news/2026/07/oracle-cloud-free-tier-limits/
- HF forum on Docker Spaces being paid: https://discuss.huggingface.co/t/new-free-accounts-cannot-create-cpu-basic-gradio-spaces-only-zerogpu-available/177629
- GitHub discussion on DigitalOcean leaving the pack: https://github.com/orgs/community/discussions/201240
- Koyeb acquisition coverage: e.g. https://www.srvrlss.io/provider/koyeb/
