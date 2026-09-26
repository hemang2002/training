# Day 5 — "Estimate It": Fermi Problems for ML Engineers

**Ten estimation problems with worked solutions.** The goal is to reason to the right **order of magnitude** within minutes, with explicit assumptions. This is how engineers size systems before building them and how they spot numbers that "smell wrong".

## How to run this block (30 min)

- Teams of 3–4, **4 minutes per problem**, paper only (a phone calculator is allowed, search is not).
- Each team writes down: **assumptions → calculation → answer with units**.
- Scoring: within 2× of the reference = 3 pts, 3× = 2 pts, 10× = 1 pt, **plus up to 2 pts for the clearest reasoning** (reasoning beats lucky numbers).
- Core set if short on time: **F1, F3, F4, F5**. The others are optional extras.

## Handy numbers (put these on the board)

| Quantity | Value |
|---|---|
| Seconds per day | 86,400 (≈ 10⁵) |
| Hours per month | ≈ 730 |
| fp32 / fp16 / int8 | 4 / 2 / 1 bytes per value |
| MobileNetV2 | ≈ 3.5M params, ≈ 0.3 GMACs @ 224 px |
| ResNet18 / ResNet50 | ≈ 11.7M / 25.6M params; ≈ 1.8 / 4.1 GMACs @ 224 px |
| Faster R-CNN MobileNetV3-Large FPN (torchvision) | ≈ 19M params |
| 1 MAC | 2 FLOPs (multiply + add) |
| Compute scales with | pixels, so (new side / old side)² |
| Light in fibre | ≈ 200,000 km/s (≈ 5 µs per km) |
| 224×224 JPEG | ≈ 10–30 KB; phone photo (12 MP) ≈ 2–5 MB |
| 1 Gbps | ≈ 125 MB/s |

> Cloud prices below are **illustrative and rounded**. They change and vary by provider and region, so always check current pricing. The *method* is what matters.

---

## F1 — How big is the model file?

**Problem:** MobileNetV2 has about 3.5M parameters. Estimate its size in FP32, FP16 and INT8. Do the same for ResNet18 and ResNet50. Why is the real INT8 file not *exactly* 4× smaller?

**Solution:** size ≈ params × bytes per param.

| Model | Params | FP32 (×4 B) | FP16 (×2 B) | INT8 (×1 B) |
|---|---|---|---|---|
| MobileNetV2 | 3.5M | **≈ 14 MB** | ≈ 7 MB | ≈ 3.5 MB |
| ResNet18 | 11.7M | ≈ 47 MB | ≈ 23 MB | ≈ 12 MB |
| ResNet50 | 25.6M | ≈ 102 MB | ≈ 51 MB | ≈ 26 MB |
| Faster R-CNN MNv3-FPN | ≈ 19M | ≈ 76 MB | ≈ 38 MB | ≈ 19 MB |

**Why not exactly 4×:**
- quantization parameters (scale and zero-point per tensor or per channel);
- biases are often kept as int32 or fp32;
- some layers are left in fp32 (first or last layer, or unsupported ops);
- graph and metadata overhead in ONNX or TorchScript.

Real INT8 files are typically **3–3.9× smaller**.

**Discussion:** the file size is *not* the runtime memory. Add activations, runtime libraries and buffers (see F2 and Q6.9). And a smaller file only means *faster* inference if the kernels actually run in INT8 (Scenario 6).

**Variation:** a 7B-parameter LLM in FP16 is about 14 GB and in 4-bit about 3.5 GB. Can it fit on a 16 GB laptop GPU? And on the 12 GB Oracle free VM (Always Free Arm allowance since June 2026), on CPU?

---

## F2 — How much GPU memory does training need?

**Problem:** estimate GPU memory to train ResNet18 at 224×224, batch size 64, FP32, with Adam.

**Solution**

1. **Static memory (per parameter):** weights 4 B + gradients 4 B + Adam m and v 8 B = **16 B per parameter**. 11.7M × 16 B ≈ **190 MB**.
2. **Activations saved for backward (per image):** add up the feature maps each layer stores (conv output, BN output, ReLU output).
   - Stem: 64×112×112 ≈ 0.8M values, × about 3 saved tensors ≈ 2.4M
   - Stage 1 (56×56×64 ≈ 0.2M per tensor, about 12 tensors) ≈ 2.4M
   - Stage 2 (28×28×128 ≈ 0.1M) ≈ 1.2M. Stage 3 ≈ 0.6M. Stage 4 ≈ 0.3M.
   - Total ≈ 7M values × 4 B ≈ **28 MB per image**.
3. Batch 64: 64 × 28 MB ≈ **1.8 GB**.
4. Add the CUDA context (about 300–500 MB), cuDNN workspace and allocator fragmentation.

**Answer:** about **2.5–3.5 GB**. Activations dominate (about 10× the weight-related memory).

**Discussion levers:** halve the batch and activations halve. Mixed precision roughly halves activations. Gradient checkpointing trades compute for activation memory. Our Day 2 setup of **MobileNetV2 at 96 px** needs only about (96/224)² ≈ 0.18× the activation memory per image of the same network at 224 px, which is why it fits on free Colab or Kaggle GPUs easily.

**Variation:** why does *inference* of the same model at batch 1 need only about 50–100 MB beyond the weights? (No stored activations, no gradients or optimizer states.)

---

## F3 — How many requests per second, and how many pods?

**Problem:** your API must serve **1 million image classifications per day**. How many req/s at average and at peak? If one request takes 25 ms of CPU time on 1 vCPU (INT8 ONNX MobileNetV2 + image decode + HTTP), how many 1-vCPU pods do you need?

**Solution**

1. Average rate: 10⁶ / 86,400 ≈ **11.6 req/s**.
2. Traffic is not flat. With a peak/average factor of about 4 (daytime peak), **peak ≈ 46 req/s**. Take 50 req/s to be safe.
3. Capacity per 1-vCPU pod: 1 / 0.025 s = 40 req/s at 100% utilization. Target 60–70% utilization to keep queueing latency low (Q8.9), so about **26 req/s per pod**.
4. Pods at peak: 50 / 26 ≈ 1.9, so **2 pods**, plus 1 for redundancy and rolling updates: **3 pods**.
5. Little's law sanity check: in-flight requests = λ × W = 50 × 0.025 ≈ 1.25, so concurrency is low and a single worker per pod is fine.

**Answer:** about 12 req/s on average, about 50 req/s at peak, **2–3 small pods** (HPA min 2, max about 5).

**Discussion:** a million per day *sounds* big but is small for an optimized CNN on a CPU. What would change with a **ResNet50 FP32** at about 150 ms per request? (6× the service time, so about 12–15 pods.) Here quantization and model choice translate directly into money.

---

## F4 — CPU or GPU: what does 1M predictions/day cost per month?

**Problem:** using F3, compare the monthly cost of serving on CPUs against a single always-on GPU instance. Assumptions: a vCPU costs about **$0.03–0.05/hour** on demand; an entry-level GPU instance (T4-class GPU + host VM) costs about **$0.50–0.75/hour**; the GPU can do about 1,000+ img/s with batching for this model.

**Solution**

| | CPU (3 × 1 vCPU pods, or one 4-vCPU VM) | GPU (1 × T4-class instance) |
|---|---|---|
| Hourly | 4 vCPU × $0.04 ≈ $0.16 | ≈ $0.60 |
| Monthly (× 730 h) | **≈ $115** | **≈ $440** |
| Utilization at peak | ≈ 60–70% | 50 / 1000 ≈ **5%** |
| Cost per 1M predictions | $115 / 30 ≈ **$3.8** | $440 / 30 ≈ **$15** |
| Redundancy | Easy (several small pods) | Needs a 2nd GPU instance for HA, so ≈ $880 |

**Answer:** the CPU is about **4× cheaper** (about 8× with GPU redundancy) at this volume. The GPU mostly sits idle.

**Break-even:** a GPU instance at about $440/month equals about 11 vCPUs at $0.04/hour, which is about 11 × 26 ≈ **290 req/s sustained** with our per-core numbers. Above roughly **25M predictions/day** (sustained, batchable), or with much heavier models (ViT-L, detection at high resolution, diffusion), the GPU wins.

**Discussion:** what if the traffic is concentrated in 2 hours per day? (Autoscaling CPUs to zero or near zero widens the gap. Serverless may be cheapest.)

---

## F5 — How much bandwidth do image uploads need?

**Problem:** 1M predictions/day. Compare uploading **raw phone photos (≈ 3 MB)** with **client-side resized 224 px JPEGs (≈ 20 KB)**. What is the average bandwidth, and how long does one upload take on a 5 Mbps mobile uplink?

**Solution**

| | Raw photo (3 MB) | Resized JPEG (20 KB) |
|---|---|---|
| Data per day | 3 MB × 10⁶ = **3 TB/day** | 20 KB × 10⁶ = **20 GB/day** |
| Average bandwidth | 3×10¹² B × 8 / 86,400 s ≈ **280 Mbps** | ≈ **1.9 Mbps** |
| Peak (×4) | ≈ 1.1 Gbps | ≈ 7.5 Mbps |
| Upload time @ 5 Mbps | 3 MB × 8 / 5 Mbps ≈ **4.8 s** | ≈ **0.03 s** |
| With base64 in JSON (+33%) | 4 MB, ≈ 6.4 s | 27 KB |

**Answer:** client-side resizing cuts bandwidth by about **150×** and upload latency from seconds to tens of milliseconds. The model only needs 96–224 px anyway.

**Discussion:** ingress is usually free on clouds, but the server still pays CPU to decode 12 MP JPEGs (decoding one 12 MP JPEG can take tens of ms, *more than the inference itself*) and memory (a 12 MP RGB image is 36 MB as uint8 and 144 MB as float32). The catch: client resizing must match the training preprocessing, or be followed by server-side resizing to the exact training size.

---

## F6 — How fast *should* inference be? (latency from FLOPs)

**Problem:** estimate the single-image CPU latency of MobileNetV2 at 224 px and at 96 px on one laptop core. Assume the core peaks at about 100 GFLOP/s FP32 (AVX2: 2 FMA units × 8 floats × 2 FLOPs × 3 GHz ≈ 96 GFLOP/s), and that small or depthwise layers reach only about 20–30% of peak.

**Solution**

1. Work at 224 px: 0.3 GMACs × 2 = **0.6 GFLOP**.
2. Effective throughput: about 25 GFLOP/s.
3. Latency ≈ 0.6 / 25 ≈ **24 ms** (1 core, FP32).
4. At 96 px: compute scales by (96/224)² ≈ 0.18, so ≈ 0.11 GFLOP, giving **≈ 4–5 ms**.
5. With 4 threads, maybe 2.5–3× faster (not 4×: parallel overheads, memory bandwidth). With INT8 and good kernels, maybe another 1.5–3×.

**Answer:** about 20–30 ms at 224 px and about 4–6 ms at 96 px on one core (FP32). Compare with your Day 2 benchmark numbers. If a measurement is **10× off**, suspect the benchmark (no warm-up, threads, included preprocessing) or the runtime (eager Python overhead, unfused ops).

**Discussion:** why is ResNet50 (4.1 GMACs, about 14× MobileNetV2's MACs) often only about 3–5× slower on a GPU? (Depthwise convs have low arithmetic intensity; GPUs favour dense, regular compute. FLOPs are not latency.)

---

## F7 — How long does it take to scale up? (and how many requests suffer)

**Problem:** traffic jumps from 20 to 80 req/s in seconds. Each pod handles about 26 req/s (F3), and you have 1 pod. How long until new pods serve traffic with (a) the **2 GB torch image** and (b) the **300 MB ONNX Runtime image**? How many requests are delayed or dropped meanwhile?

**Assumptions:** HPA detection about 30 s (metrics scrape + sync). The node pulls at about 50 MB/s (compressed image about 40% of its size). Extraction takes about as long as the pull. Container start plus imports plus model load: torch about 10 s, ORT about 2 s. The image is not cached on the node.

**Solution**

| Step | Torch image (2 GB) | ORT image (300 MB) |
|---|---|---|
| HPA notices | 30 s | 30 s |
| Pull (~800 MB vs ~120 MB compressed @ 50 MB/s) | 16 s | 2.4 s |
| Extract | ≈ 16 s | ≈ 2.4 s |
| Start + imports + model load | ≈ 10 s | ≈ 2 s |
| Readiness probe passes | ≈ 5 s | ≈ 5 s |
| **Total** | **≈ 77 s** | **≈ 42 s** |

Capacity deficit: 80 − 26 = **54 req/s** unserved.
- Torch: 54 × 77 ≈ **4,200 requests** queued, slowed or failed.
- ORT: 54 × 42 ≈ **2,300 requests**.

**Answer:** slim images cut about 35 s, and the HPA reaction time is now the biggest remaining piece. Mitigations: higher `minReplicas` (headroom), a lower CPU target, pre-pulled images on nodes, scaling on RPS or queue length, faster readiness.

**Discussion:** on a *new node* (cluster autoscaler), add 1–3 minutes of VM boot, which is why headroom matters.

---

## F8 — Can the dataset fit in memory?

**Problem:** (a) How much memory does CIFAR-10's training set take as uint8 at 32×32? (b) What if you pre-resize it to 96×96 and store it as float32 tensors "to speed things up"? (c) If the GPU trains at 2,000 img/s, how many CPU data-loader workers do you need if one worker decodes, augments and resizes 400 img/s?

**Solution**
- (a) 50,000 × 32 × 32 × 3 B = **153.6 MB**. It fits anywhere.
- (b) 50,000 × 96 × 96 × 3 × 4 B = **5.53 GB**, 36× larger (9× the pixels × 4× the bytes). It may not fit in Colab RAM and wastes memory bandwidth.
- (c) 2,000 / 400 = **5 workers**. Fewer, and the GPU starves (utilization drops, epochs slow down).

**Answer:** store compactly (uint8, small), and resize and normalize **on the fly** in the DataLoader or on the GPU. The data pipeline is often the real bottleneck.

**Discussion:** FashionMNIST: 60,000 × 28 × 28 B ≈ 47 MB. ImageNet-1k: about 1.28M JPEGs ≈ 140–150 GB. Why can't that be loaded into RAM, and what does that imply for the data-loading design?

---

## F9 — Energy and carbon: what dominates?

**Problem:** a CPU server drawing **100 W** serves 1M predictions/day at about 20 ms of CPU time each. (a) Energy per prediction? (b) Daily energy of the predictions vs keeping the server on 24/7? (c) Daily CO₂ at a grid intensity of about **0.7 kg CO₂/kWh** (roughly India's grid average)?

**Solution**
- (a) Crude upper bound: attribute the full 100 W to each prediction for 20 ms, giving 100 W × 0.02 s = **2 J per prediction**.
- (b) Predictions: 10⁶ × 2 J = 2 MJ ≈ **0.56 kWh/day**. Server always on: 100 W × 24 h = **2.4 kWh/day**. **Idle time dominates**: about 75% of the energy does no useful work.
- (c) 2.4 kWh × 0.7 ≈ **1.7 kg CO₂/day** ≈ 0.6 t/year for this one server.

**Answer:** for small models, **utilization and idle time matter more than per-inference efficiency**. Autoscaling, right-sizing and consolidation cut energy more than shaving 5 ms. For huge models at huge scale, per-inference efficiency (INT8, distillation) dominates (Q10.7).

**Variation:** training MobileNetV2 on CIFAR-10 for 1 hour on a 70 W T4: 0.07 kWh. How many days of serving equal that? (Less than one hour of the always-on server.)

---

## F10 — How much logging and monitoring data?

**Problem:** 1M predictions/day. You log one JSON line per request (timestamp, latency, version, top-3 classes and scores, input size, hash) of about **500 B**. You also consider storing **every input image** (20 KB, resized) for retraining. Estimate storage per month and a rough cost (log ingestion about **$0.50/GB**; object storage about **$0.02/GB-month**).

**Solution**

| | Per day | Per month | Rough monthly cost |
|---|---|---|---|
| Request logs (500 B) | 0.5 GB | 15 GB | ingestion 15 × $0.50 ≈ **$7.5** |
| All images (20 KB) | 20 GB | 600 GB (grows every month) | month 1 ≈ $12, month 12 ≈ **$144** (7.2 TB) |
| 1% sampled images + all low-confidence ones (≈ 3%) | 0.6 GB | 18 GB | ≈ **$0.36** (+ labelling cost) |

**Answer:** logs are cheap. Storing every image grows linearly and has **privacy and legal implications** (consent, retention, deletion requests; Q10.5). **Sampling** (random + uncertain + user-flagged) gives most of the retraining value at a few percent of the cost and risk.

**Discussion:** what should the sampling strategy be to avoid a biased retraining set? (Always include a *random* share, not only low-confidence cases; see the feedback loops in Q9.9.)

---

## Debrief questions

1. Which assumption changed the answer the most in each problem? (Sensitivity analysis is a core skill.)
2. Which estimates surprised you? (Common ones: activations dominate training memory; the GPU is idle at 1M/day; idle energy dominates; uploads dwarf inference.)
3. Where would you *measure* instead of estimating, and with what tool? (`torch.cuda.max_memory_allocated`, a benchmark script, `hey`/`locust`, `docker stats`, cloud billing reports.)
