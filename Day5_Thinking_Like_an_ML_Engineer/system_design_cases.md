# Day 5 — System Design Mini-Cases

**Five case studies.** Each one asks students to turn vague requirements into an end-to-end ML system using what they learned on Days 1–4.

## How to run this block (50 min)

| Phase | Time | What happens |
|-------|------|--------------|
| Brief | 3 min | Each group (4–5 people) gets one case card: **Scenario + Requirements + Constraints** only |
| Clarify | 3 min | Groups may ask the trainer (the "client") up to **3 clarifying questions**. Good questions earn points |
| Design | 20 min | On a flipchart or whiteboard: architecture diagram, model and optimization choice, deployment target, monitoring, top 3 risks |
| Present | 4 min / group | The presenter walks through the design |
| Critique | 2 min / group | The other groups ask one question each; the trainer throws one **curveball** |
| Debrief | 5 min | The trainer contrasts designs against the reference sketch; there is no single right answer |

**Every design must answer these six questions** (put them on the board):
1. What is the **input**, the **output**, and the **latency or throughput** target?
2. Which **model**, and which **optimization** (quantization, pruning, distillation, resolution), and *why*?
3. **Where** does inference run (edge, VM, PaaS, k8s, serverless), and how is it packaged?
4. How does it **scale and fail** (what happens under a spike, a crash, a lost network)?
5. How do you **monitor** it and know when it is wrong (drift, calibration, feedback)?
6. What are the **risks** (cost, privacy, fairness, safety), and how are they mitigated?

**Rubric (10 pts):** requirements coverage 2 · architecture clarity 2 · justified trade-offs 2 · monitoring and feedback 2 · presentation 2.

---

## Case 1 — Defect detector on a factory edge device

### Scenario
A mid-sized factory makes metal brackets. Parts pass under a camera on a conveyor. They want to automatically **reject defective parts** (scratches, cracks, missing holes). The line is in a basement with **no internet access** for security reasons. IT gave them a small ARM single-board computer.

### Requirements
- Classify each part as OK or DEFECT (optionally the defect type), and trigger a pneumatic reject arm through GPIO.
- Throughput: **5 parts/second**. The decision is needed **< 150 ms** after the trigger.
- Missing a defect (false negative) is **10× more costly** than wrongly rejecting a good part.
- Operators must be able to see recent decisions and override them.

### Constraints
- Device: quad-core ARM Cortex-A72 class, **2 GB RAM**, 32 GB SD card, no GPU.
- **No internet, ever.** Model updates arrive on a USB stick once per quarter.
- Training data: 3,000 OK images and **only 150 defect images**.
- Camera: 1280×720 RGB, fixed position, controlled lighting (mostly).

### Guiding questions
1. With only 150 defect images, is this a classification problem or an **anomaly detection** problem?
2. Budget the 2 GB RAM. What takes memory besides the model?
3. Which Day 2 optimization fits an ARM CPU best? Which PyTorch quantization engine or which ONNX Runtime build?
4. Do you need Docker? Kubernetes? What gives the most reliable operation on one box?
5. How do you get feedback and drift signals with no internet?
6. How do you choose the decision threshold given the 10:1 cost ratio?
7. What happens if the device crashes mid-shift?

### Reference solution sketch

```
   ┌──────────┐  trigger   ┌───────────────────────────── Edge device (ARM, 2 GB) ──────────────────────────┐
   │ Conveyor │──────────► │  capture.py ──► preprocess (crop ROI, resize 224, normalize)                     │
   │  sensor  │            │        │                                                                         │
   └──────────┘            │        ▼                                                                         │
   ┌──────────┐  frames    │  ONNX Runtime (INT8, 1–2 threads)  ──► p(defect) ──► threshold τ ──► GPIO reject │
   │  Camera  │──────────► │        │                                                  │                      │
   └──────────┘            │        ▼                                                  ▼                      │
                           │  SQLite log (ts, score, decision, version) + JPEG of every reject & 1% random OK │
                           │        │                                                                         │
                           │        ▼                                                                         │
                           │  Local web dashboard (Flask/Streamlit on LAN) ◄── operator override buttons      │
                           │  systemd services + hardware watchdog; read-only root FS                        │
                           └──────────────────────────────────────────────────────────────────────────────────┘
                                         ▲ quarterly USB: signed model bundle (model.onnx + config + checksum)
                                         │ ▼ export of labelled overrides + samples for retraining
                                   Offline training workstation (GPU) at head office
```

**Model and optimization**
- Backbone: **MobileNetV2** (about 3.5M params) pretrained, fine-tuned at 224 px on the **cropped region of interest**. The fixed camera means cropping is cheap and removes background.
- The data imbalance (3,000 vs 150) calls for a class-weighted loss, heavy augmentation (flips if symmetric, brightness, blur), and possibly an **anomaly-detection** backup (features of OK parts + distance score) to catch defect types never seen in training.
- **Static INT8 PTQ, or QAT if PTQ loses more than 1% recall**, exported through ONNX, run with ONNX Runtime on ARM (or PyTorch with the `qnnpack` engine). INT8 cuts the model to about 3.5 MB and uses ARM int8 dot-product paths where available.
- Latency estimate: MobileNetV2 at 224 is about 0.3 GMACs. At an effective 3–10 GMAC/s on 2 A72 cores, that is roughly **30–100 ms**. It fits in 150 ms, but the whole pipeline (capture + preprocess + inference + GPIO) must be measured. Cropping and 160 px input are fallbacks.

**Memory budget (2 GB)**

| Item | Approx. |
|---|---|
| OS + services | 300–400 MB |
| Python + numpy + ORT | 100–150 MB |
| Model (int8) + ORT arena | 10–50 MB |
| Frame buffers (720p RGB ≈ 2.8 MB each × a few) | < 50 MB |
| Dashboard | 50–100 MB |
| **Headroom** | > 1 GB |

**Deployment:** no Kubernetes. It is one box with no network, so an orchestrator adds failure modes without benefit. Use **systemd** units (auto-restart), a hardware watchdog, and a read-only root filesystem (the SD card survives power cuts). Docker (arm64 image) is optional, useful for reproducible builds. Updates are a **signed bundle**: verify the checksum, keep the previous model for instant rollback, and run a smoke test on 20 stored golden images before switching.

**Threshold:** choose τ on validation data to minimize expected cost = 10·FN + 1·FP (usually a low threshold, so high recall). Calibrate the scores (temperature scaling) so τ stays meaningful.

**Monitoring without internet:** local dashboard with reject rate per hour (a sudden jump means a lighting change or a new defect type), score histogram, mean image brightness (drift), operator override rate. Store all rejects and a 1% random sample of OKs. Exported quarterly, these become new labelled data, and random sampling of OKs avoids the feedback-loop problem (Q9.9).

**Failure mode:** if inference fails or the watchdog trips, **fail safe**: reject everything or stop the line and alert, depending on what the business agrees. Never silently pass parts.

**Key trade-offs to discuss:** classifier vs anomaly detector; INT8 accuracy vs latency; fail-open vs fail-closed; simplicity (systemd) vs tooling (k3s).

**Curveballs:** *"The factory adds a second line. Same device?"* (Throughput doubles, so benchmark it; batch 2 frames or add a device.) *"Lighting changes in summer."* (Brightness drift monitoring, augmentation, controlled enclosure.) *"A new defect type appears."* (The anomaly score catches it; the quarterly retrain adds it.)

---

## Case 2 — Image classifier for 10k users with spiky traffic, on a student budget

### Scenario
Your team's CIFAR-style classifier app (Day 3 API + Streamlit) got featured on a popular social account. There are **10,000 registered users**. Traffic is normally quiet, but **spikes 50×** for a few hours after posts. You have **$0–20 per month** and GCP/Azure student credits.

### Requirements
- p95 latency < 1 s end-to-end during normal load; degrade gracefully (not crash) during spikes.
- Normal traffic about 50k predictions/day. Spike peak up to **30 req/s** for 1–2 hours.
- Show model version; support rolling out a new model without downtime.

### Constraints
- Budget: ideally ≤ $20/month after free tiers.
- Team of 3 students, part-time: minimal operations work.
- The current image is torch-based, 2.1 GB, and takes 45 s to cold start.

### Guiding questions
1. Average vs peak req/s: what capacity do you need, and for how long?
2. Always-on VM vs scale-to-zero serverless vs free-tier PaaS: cost and cold start for each?
3. What single change most improves cold start and cost? (Hint: Day 2 + Day 4.)
4. How do you protect the budget from abuse or accidental runaway scaling?
5. Where can you cut latency apart from the model? (Client-side resize, region.)
6. How do you roll out model v2 safely?

### Reference solution sketch

```
  Users ──► Streamlit UI (Render free / Streamlit Community Cloud)
               │ client-side resize to ≤ 256 px JPEG (~20 KB)
               ▼ HTTPS
        ┌─────────────────────────────── Serverless containers (e.g. Cloud Run) ─────────────────────────┐
        │  API container: gunicorn (1–2 workers) + ONNX Runtime INT8 MobileNetV2 (image ~300 MB)         │
        │  min instances: 0 (or 1 during announced campaigns), max instances: 5  ◄── budget cap            │
        │  concurrency per instance tuned by load test (e.g. 4–8)                                          │
        └──────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                   │ structured logs (latency, version, class, confidence)
                                                   ▼
                                  Cloud logging + budget alerts ($5/$10/$20) + uptime check

  Alternative (fixed $0): Oracle Always Free Ampere VM (2 OCPU/12 GB since June 2026) + k3s + HPA (min 2, max 6 pods), arm64 image
```

**Capacity math**
- Normal: 50k/day is about **0.6 req/s**, so any single instance idles.
- Spike: 30 req/s. INT8 MobileNetV2 at about 10–20 ms per image per vCPU gives roughly 50–100 req/s per vCPU for inference alone. HTTP and decode overhead cut that in half, so **1–2 vCPUs** handle the spike with headroom. The real challenge is **scaling fast enough**, not raw compute.

**Key decisions**
1. **Switch to the ONNX Runtime INT8 image (about 300 MB)**. Cold start drops from about 45 s to a few seconds. It is cheaper per request and the single most valuable change.
2. **Scale-to-zero serverless** fits "quiet + spiky": you pay nearly nothing when idle, and the free monthly allowances of Cloud Run-style services often cover this volume (check current limits). Set **max instances** as a hard cost cap.
3. **Client-side resize** cuts uploads from 3–4 MB phone photos to about 20 KB. That means faster uploads, less bandwidth and less server decode CPU. (The server must still apply exactly the training preprocessing.)
4. **Rate limiting** per user or IP, plus an upload size limit, protects the budget and prevents abuse.
5. **Rollout:** deploy a new revision, send 10% of traffic to it (traffic splitting), watch errors and the agreement rate, then go to 100%. Roll back in one command.

**Comparison**

| Option | Monthly cost (est.) | Cold start | Ops effort | Spike handling |
|---|---|---|---|---|
| Render/HF free | $0 | 30–60 s after idle | Very low | Poor (single instance) |
| Serverless containers | ≈$0–10 | 2–10 s with a slim image | Low | Good (auto, capped) |
| Always-on small VM | ≈$15–30 | None | Medium | Fixed capacity |
| Oracle Free VM + k3s | $0 | None | Higher (you run k8s) | Good within 2 OCPU |
| Managed K8s | $70+ (control plane + nodes) | None | High | Excellent (overkill) |

**Monitoring:** p95 latency, 5xx rate, instance count, cost per day (budget alerts), prediction distribution (a viral post may bring new kinds of images, which is drift).

**Curveballs:** *"A bot sends 1M requests overnight."* (max instances, rate limiting, auth, budget alerts: autoscaling turns DoS into a billing attack.) *"Users in India complain about latency."* (Region choice matters more than the model; see Q8.7.) *"Credits expire next month."* (Move to the Oracle free VM; that is why multi-arch images help.)

---

## Case 3 — Hospital chest X-ray triage with privacy constraints

### Scenario
A hospital's radiology department has a backlog. They want a model that **prioritizes the reading queue**: studies likely to contain urgent findings (for example pneumothorax) move to the top so radiologists read them first. The model **does not diagnose**.

### Requirements
- Score each new chest X-ray within **2 minutes** of acquisition and reorder the worklist.
- High **sensitivity** for urgent findings. Radiologist capacity limits how many studies can be flagged urgent.
- Radiologists see a score and an explanation aid (heat-map) but make every decision.
- Full **audit trail**: which model version scored which study, and when.

### Constraints
- **Patient data must not leave the hospital network.** No public cloud.
- Hospital IT provides one on-prem server with one mid-range GPU (or CPU only), running Linux and Docker.
- Integration with PACS (the imaging archive) through DICOM.
- Volume: about **400 X-rays/day**, mostly in daytime.
- Regulatory: likely a medical device (software); changes require validation.

### Guiding questions
1. 400/day: how much compute do you actually need? Is a GPU necessary?
2. Would you quantize this model aggressively? What is the risk trade-off compared with Case 1?
3. Why is **calibration** essential here? (Q9.4.)
4. How do you choose the threshold, and who chooses it?
5. How would you detect when a new X-ray machine (different vendor) degrades performance?
6. How do you avoid automation bias (Q10.9)?
7. How do you update the model under regulatory constraints?

### Reference solution sketch

```
 ┌────────── Hospital network (no internet egress for PHI) ────────────────────────────────────────────────┐
 │                                                                                                        │
 │  X-ray modality ──DICOM──► PACS ──DICOM C-STORE / DICOMweb──► Inference gateway (listener)              │
 │                                                                   │ de-identify metadata for logs      │
 │                                                                   ▼                                    │
 │                                   Queue (Redis/RabbitMQ) ──► Inference worker(s) (Docker, GPU/CPU)       │
 │                                                                   │  model: DenseNet121/ResNet50,       │
 │                                                                   │  FP32/FP16, temperature-scaled     │
 │                                                                   ▼                                    │
 │                              Results DB (study UID, score, calibrated prob, model ver, ts)  ─► Audit log│
 │                                                                   │                                    │
 │                                                                   ▼                                    │
 │                         Worklist integration (priority flag) + viewer overlay (Grad-CAM, "aid only")   │
 │                                                                                                        │
 │  Monitoring: score distribution per scanner/vendor, flag rate, radiologist agreement, latency, errors  │
 └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
          ▲ model updates: validated offline on local retrospective data, signed image, versioned release
```

**Key decisions**
- **Compute:** 400/day is about **one study every 3–4 minutes** on average, a trivial load. A CPU can do it (a ResNet50-class model at 512 px takes about 1 s on CPU). A GPU gives headroom for higher resolution or ensembles. **No Kubernetes needed.** Docker Compose with restart policies and a queue gives robustness.
- **Optimization is *not* the priority:** latency needs are loose (2 minutes) and errors are costly. Use FP32, or FP16 on the GPU (negligible accuracy impact). **Avoid aggressive INT8 or pruning** unless validated per finding and per subgroup, because compression can hurt rare cases disproportionately (Q10.6).
- **Calibration and threshold:** temperature-scale on local validation data. Choose the threshold with clinicians, using the ROC curve and radiologist capacity (for example flag the top 15% while keeping sensitivity ≥ 95% for pneumothorax). **Recalibrate per site**, since prevalence and scanners differ.
- **Privacy:** on-prem only. PHI stays in the hospital. Logs hold study UIDs, not images or names. Role-based access. Encryption at rest. Retention policy. Compliance with HIPAA, GDPR or India's DPDP Act as applicable.
- **Human in the loop:** the output **reorders** the queue and never auto-clears a study. The UI shows a calibrated likelihood and a heat-map labelled "aid only". Audit radiologist overrides.
- **Monitoring and drift:** score distribution and flag rate **per scanner vendor or site**. A step change means the new machine or protocol needs local validation. Periodic sampling with radiologist labels gives true sensitivity.

**Trade-offs:** sensitivity vs radiologist workload; explainability aids vs automation bias; model improvements vs revalidation burden.

**Curveballs:** *"A new portable X-ray machine is added in the ICU."* (Distribution shift: shadow-mode evaluation before trusting its scores.) *"Can we use cloud GPUs for retraining with de-identified data?"* (Possible with de-identification, agreements and governance. Discuss re-identification risk.) *"The model flags 40% of studies on Monday."* (Drift or a bug: check scanner, preprocessing and version; fall back to the normal worklist.)

---

## Case 4 — Real-time pedestrian detection on a drone

### Scenario
A search-and-rescue team wants drones that **detect people on the ground** in real time and send their GPS positions to a ground station. The Penn-Fudan Faster R-CNN from the Day 2 demo is the starting point.

### Requirements
- Onboard detection at **≥ 15 FPS** (ideally 30) from a 1080p camera.
- Send detections (GPS + confidence + small thumbnail) to the ground station over a **weak radio link** (about 1 Mbps, intermittent).
- High **recall**. Missing a person is the worst outcome.
- Flight time should not drop much: a tight **power budget (≈10–15 W)** for compute.

### Constraints
- Onboard computer: embedded GPU module (Jetson-class, 8 GB shared memory), or a weaker ARM CPU-only board as a fallback.
- People seen from **30–100 m altitude** look **tiny** (10–30 px tall in 1080p) and from above. Penn-Fudan is street-level.
- The radio link drops for tens of seconds at a time.

### Guiding questions
1. Can Faster R-CNN MobileNetV3 hit 15 FPS on this hardware? What did the Day 2 optimizations (resolution, proposals) show?
2. Reducing resolution speeds things up. What does it do to **tiny** people? (Q4.1.)
3. One-stage vs two-stage detector here? (Q4.3.)
4. Is Penn-Fudan the right training data? What data do you need?
5. How do you use the time between frames: detect every frame, or detect + track?
6. What goes over the radio, and what happens when the link drops?
7. Privacy: the drone films everyone. What are the obligations?

### Reference solution sketch

```
 ┌──────────────────────────── Drone (embedded GPU, ~10–15 W) ─────────────────────────────────┐
 │ Camera 1080p@30 ─► resize/tiles (e.g. 2×2 tiles of 640 px) ─► Detector (one-stage, TensorRT│
 │                                                               FP16/INT8, 640 px)           │
 │                          ▲                                            │ boxes + scores      │
 │                          │ detect every 2–3 frames                    ▼                     │
 │                          └───────────────────────  Multi-object tracker (SORT/ByteTrack)   │
 │                                                              │ tracks (stable IDs)          │
 │                                                              ▼                              │
 │                   Geo-projection (drone GPS + altitude + gimbal angle) ─► detection events │
 │                                                              │                              │
 │                         Local store-and-forward queue (SQLite) ◄── full-res crops on SD    │
 └──────────────────────────────────────────────────────────────┼──────────────────────────────┘
                                                                │ radio (~1 Mbps, lossy): JSON event
                                                                │ (≈200 B) + 64×64 thumbnail (≈3 KB)
                                                                ▼
                                   Ground station: map UI, operator confirms / dismisses, requests full crop
```

**Model and optimization**
- The Faster R-CNN MobileNetV3-FPN demo is likely **too slow at the resolution tiny objects need**. Two-stage heads cost time per proposal, and raising the resolution to see small people multiplies backbone cost. Use it as a baseline, not the target.
- Pick a **one-stage detector** (SSDlite or a small YOLO-family model) at about 640 px, exported via ONNX to **TensorRT FP16** (on an embedded GPU FP16 is nearly free in accuracy; INT8 with calibration if still too slow, validated on small-object recall).
- **Small objects:** tiling (split the frame into overlapping tiles), or a higher input resolution with a lighter backbone. Keep the anchor sizes/strides able to detect about 10 px objects. Resolution is the biggest lever and the biggest risk.
- **Detect + track:** run the detector every 2–3 frames and track in between. Tracks smooth out flicker, give stable IDs (no duplicate alerts), and roughly halve compute.
- **Data:** Penn-Fudan (street-level, frontal) doesn't match aerial views. Use aerial person datasets and your own flights. Augment scale, rotation (any orientation from above), motion blur and lighting.

**Communication design:** send **events, not video**. JSON (about 200 B) plus a small thumbnail (about 3 KB) per new track, which fits even at 1 Mbps. **Store-and-forward** queue when the link drops. The operator can ask for a higher-resolution crop.

**Power and latency budget:** measure FPS per watt. Use the module's power modes. The FP16/INT8 engine plus frame skipping keeps the load within budget.

**Monitoring:** onboard FPS, temperature and throttling, detection count per minute; after the flight, compare operator-confirmed vs dismissed detections to estimate precision, and add missed people found by ground teams to the training data.

**Ethics and regulation:** only operate under aviation rules and a rescue mandate. Keep data minimal (thumbnails of people, retained only for the mission). No face recognition. Document the limitations (night, dense forest canopy).

**Curveballs:** *"It must work at night."* (Thermal camera: a new modality, new data, and a different pretrained model.) *"The embedded GPU is unavailable; only a CPU board."* (A lower-resolution region of interest, INT8 ORT, heavy frame skipping, accepting a lower FPS, or offloading when the link allows.) *"Birds and rocks trigger false alarms."* (Operator confirmation loop, harder negatives in training, track-length filtering.)

---

## Case 5 — A GAN-based avatar generator app

### Scenario
A startup wants an app where users upload a selfie and receive **stylized avatars** (cartoon, anime, oil painting). They trained an image-to-image GAN generator (the conditional big brother of the Day 2 DCGAN). Launch is in 6 weeks and marketing expects **50,000 sign-ups in launch week**.

### Requirements
- Upload a selfie and receive 4 avatars in **< 30 s**. A "your avatars are ready" notification is acceptable.
- Handle a launch-day burst (**10× normal**) without the app falling over.
- Keep cost per avatar low enough for a freemium model (target **< $0.002 per avatar**).
- Delete user photos after processing unless the user opts in.

### Constraints
- Teacher generator: 45M params, about 180 ms per 512×512 image on a T4 GPU at FP16, about 4 s on CPU.
- A small team, familiar with Docker and some Kubernetes.
- Legal is worried about **deepfakes, consent and minors**.

### Guiding questions
1. Synchronous API or **asynchronous job queue**? Why?
2. GPU or CPU for inference? What about **distilling** the generator (Q4.11) and quantizing it (Q4.10)?
3. How do you absorb a 10× burst without paying for 10× GPUs all week?
4. How do you estimate cost per avatar?
5. How do you evaluate quality after optimization? (FID, human review, Q4.8.)
6. What abuse scenarios exist, and what product and technical mitigations would you add?

### Reference solution sketch

```
 Mobile/Web app ──► API (FastAPI/Flask, CPU, autoscaled) ──► validate + face check + moderation
      ▲                     │  1. store upload in object storage (TTL 24 h, encrypted)
      │                     │  2. enqueue job {user, style, seed(s), image URI}
      │                     ▼
      │               Job queue (Redis/SQS/Pub-Sub) ──► GPU workers (k8s Deployment, KEDA scale on queue length)
      │                                                    │ distilled generator, FP16 (TensorRT/ORT CUDA)
      │                                                    │ batch 4–8 images per call
      │                                                    ▼
      │                                        Watermark + C2PA provenance ─► object storage (results)
      │                                                    │
      └──────── push notification / polling ◄──────────────┘  delete original selfie after processing
 Monitoring: queue length & age, GPU utilization, cost/avatar, FID on fixed seeds, moderation hits, complaints
```

**Key decisions**
- **Asynchronous queue:** generation takes seconds, not milliseconds, and bursts are expected. The queue absorbs the burst, and users see "ready in about 20 s" instead of timeouts. Workers scale on **queue length** (KEDA or custom metrics), not CPU.
- **Distillation, then FP16:** distill the 45M generator into a student about 4–8× smaller (supervised regression + perceptual loss, Q4.11), then FP16 on the GPU. Only use INT8 after FID and human-review checks, because generators show quantization artefacts directly (Q4.10). A CPU fallback becomes possible with the student.
- **Cost estimate (illustrative):** say the student generates 1 image in 40 ms at FP16 on a T4 with batching, so about 25 images/s, or **90,000 images/hour**. At about $0.35–0.55/hr for a T4 (varies by provider and region; spot is cheaper), that is well under **$0.0001 per avatar** at full utilization. Real cost is dominated by **idle time**, so scaling to zero or near zero off-peak matters more than raw speed. Compare the teacher at 180 ms: about 4.5× more GPU time.
- **Burst handling:** a pre-scaled GPU pool on launch day (planned), spot GPUs for extra capacity (jobs are retryable, so pre-emption is fine), maximum replica caps and a **visible queue ETA**.
- **Quality gates:** FID on a fixed seed and face set before and after every optimization; human review panel; bias check (quality across skin tones, ages, genders).

**Responsible AI (the design is incomplete without this):**
- **Consent:** terms plus a "this is me" confirmation; a liveness or face-match check against the account to stop avatars of *other* people being generated; age gating.
- **Provenance:** visible and invisible watermarks plus C2PA content credentials.
- **Data minimization:** delete selfies after processing by default; keep no training use without explicit opt-in (DPDP Act/GDPR).
- **Moderation:** block nudity and violence inputs and outputs; a report-abuse flow.
- **Training data provenance:** licensed or consented face datasets. Check for **memorization** (nearest-neighbour search of outputs against training faces).

**Curveballs:** *"Marketing wants 'real-time' previews in the camera."* (A tiny on-device distilled model at low resolution for previews, and the server for the final render.) *"Someone generates an avatar of a celebrity."* (Face-match to account holder, policy, takedown process.) *"The GPU bill tripled."* (Check idle workers, batch size, min replicas, retries in a loop, and cost per avatar per style.)

---

## Cross-case discussion (5 min)

| Question | Case 1 Edge | Case 2 Spiky web | Case 3 Hospital | Case 4 Drone | Case 5 GAN app |
|---|---|---|---|---|---|
| Dominant constraint | RAM, offline | Budget, spikes | Privacy, safety | Latency, power, tiny objects | Burst, cost, abuse |
| Best optimization | INT8 (ARM) | INT8 + slim image | Calibration (not compression) | TensorRT FP16/INT8 + tracking | Distillation + FP16 |
| Orchestration | systemd | Serverless (or k3s) | Docker Compose | Onboard process | K8s + queue autoscaling |
| Sync vs async | Sync (real-time) | Sync | Async (queue) | Streaming | Async (queue) |
| Top risk | Missed defects | Cost runaway | Automation bias, drift by scanner | Missed people | Deepfake misuse |

**Takeaway for students:** the "best model" is the one that satisfies the *constraints*. Accuracy is one requirement among latency, memory, cost, privacy, safety and operability, and the constraints differ in every case.
