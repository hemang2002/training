# 04 · Any free Linux VM + docker compose

The most "real" experience: you get a Linux server with a public IP, SSH into it, install Docker, and run the
same `docker-compose.yml` you used locally. Works on any provider's free VM — which ones are currently free
is in `cloud/README.md` (Oracle Cloud Always Free, Google Cloud free tier e2-micro, Azure for Students credit,
AWS free credits, …).

> 💸 **Cost safety first:** pick only resources labelled *Always Free* / *Free tier eligible*, set a **budget
> alert** in the billing console, and **delete the VM after the demo**. Cloud accounts can incur charges if you
> create non-free resources.

## 1. Create the VM (provider-specific, ~10 min)

| Setting | Choose |
|---|---|
| OS image | **Ubuntu 22.04 or 24.04** |
| Size | the free-eligible shape (e.g. Oracle `VM.Standard.A1.Flex` ARM or `VM.Standard.E2.1.Micro`, GCP `e2-micro` in a free region, …) |
| Disk | default (≥ 30 GB is comfortable; images need ~1 GB) |
| SSH key | upload your public key (`ssh-keygen -t ed25519` creates one) |
| Firewall | allow inbound TCP **22** (SSH) and **8501** (Streamlit). Optional: 5000 (API) |

Firewall locations:
* **Oracle:** Networking → VCN → Security List → *Add Ingress Rule* (source `0.0.0.0/0`, TCP 8501).
  **Also** open the OS firewall — Oracle's Ubuntu images ship with restrictive iptables rules:
  ```bash
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8501 -j ACCEPT
  sudo netfilter-persistent save
  ```
* **GCP:** VPC network → Firewall → create rule, target all instances, TCP 8501.
* **Azure:** VM → Networking → *Add inbound port rule* 8501.
* **AWS:** EC2 → Security Groups → inbound rule TCP 8501.

## 2. Install Docker (on the VM)

```bash
ssh ubuntu@<public-ip>            # GCP: your username; Azure: the admin user you chose

curl -fsSL https://get.docker.com | sudo sh     # official convenience script (Engine + compose plugin)
sudo usermod -aG docker $USER && newgrp docker  # run docker without sudo
docker run --rm hello-world
```

**Small VMs (1 GB RAM, e.g. e2-micro / E2.1.Micro): add swap** or builds/containers may be OOM-killed:
```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 3. Get the app onto the VM

Option A — copy the folder from your laptop (includes models):
```bash
# on your laptop, from Day4_Docker_Kubernetes_Cloud/
scp -r ml-app ubuntu@<public-ip>:~/
```
Option B — `git clone` your repo on the VM (make sure `ml-app/models/` is committed or copy it with scp).

Building **on the VM** automatically produces images for the VM's CPU architecture (ARM on Oracle A1) —
no buildx needed.

## 4. Run

```bash
cd ~/ml-app
docker compose up -d --build
docker compose ps                 # both "healthy"/"running"
curl http://localhost:5000/ready
```

Open `http://<public-ip>:8501` on your phone. 🎉

## 5. Clean up

```bash
docker compose down
```
Then **terminate the VM** in the console (and release any reserved public IP) when the demo is over.

## Talking points

* This is "IaaS": you manage the OS, updates, firewall, restarts. Compare with Hugging Face/Render ("PaaS").
* No HTTPS here (plain `http://IP:8501`). Production would add a domain + reverse proxy (Caddy/Nginx) + TLS.
* One VM = single point of failure. Kubernetes (next guide) adds self-healing — but on one VM it still dies with the VM.
