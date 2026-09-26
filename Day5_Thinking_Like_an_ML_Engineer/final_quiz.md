# Day 5 — Final Quiz

**30 questions covering Days 1–5.** Individual, **20 minutes**. Then the trainer walks through the 5 most-missed questions (5 min).

| Section | Type | Questions | Points |
|---|---|---|---|
| A | Multiple choice (one correct answer) | 1–17 | 1 each |
| B | True / False | 18–24 | 1 each |
| C | Short answer (1–3 sentences) | 25–30 | 2 each |
| | **Total** | **30** | **36** |

> Trainer tip: Section A and B work well in Kahoot or Google Forms. Collect Section C on paper and mark it during the break, or peer-mark it with the answer key.

---

## Section A — Multiple choice

**1. (Day 1)** A freshly initialized 10-class classifier gives near-uniform predictions. What cross-entropy loss do you expect at step 0?
- A) 0.0
- B) 0.693
- C) 2.303
- D) 10.0

**2. (Day 1)** How many parameters does `nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)` have (with bias)?
- A) 32
- B) 288
- C) 320
- D) 25,088

**3. (Day 2)** Why did we resize CIFAR-10 images from 32×32 to 96×96 before fine-tuning MobileNetV2?
- A) Upscaling adds detail that improves accuracy
- B) It better matches the object scale and receptive fields the pretrained filters expect, and avoids a 1×1 final feature map
- C) MobileNetV2 raises an error for inputs smaller than 96×96
- D) It reduces training time

**4. (Day 2)** PyTorch dynamic quantization (`quantize_dynamic`) gave almost no speed-up on MobileNetV2. The most likely reason is:
- A) The CPU does not support INT8
- B) Dynamic quantization targets `nn.Linear`/`nn.LSTM` by default, and MobileNetV2's compute is in convolutions
- C) The model was not pruned first
- D) Dynamic quantization only works on GPU

**5. (Day 2)** In static post-training quantization, the calibration dataset is used to:
- A) Fine-tune the weights
- B) Estimate activation ranges (scales and zero-points)
- C) Measure final test accuracy
- D) Choose which layers to prune

**6. (Day 2 / Day 5)** Research on compressed networks (e.g. Hooker et al., "What Do Compressed Deep Neural Networks Forget?") found that pruning and quantization:
- A) Always improve fairness
- B) Degrade all classes equally
- C) Can disproportionately hurt rare or underrepresented examples even when overall accuracy barely changes
- D) Have no measurable effect on any subgroup

**7. (Day 2 demo)** In torchvision's Faster R-CNN, the class label `0` is reserved for:
- A) The first object class (e.g. "person")
- B) Background
- C) Ignored or difficult objects
- D) Crowd annotations

**8. (Day 2 demo)** Lowering the number of RPN proposals kept at test time mainly speeds up:
- A) The backbone
- B) The feature pyramid network
- C) The second-stage RoI heads (RoIAlign + box head run per proposal)
- D) Image loading

**9. (Day 2 demo)** Which metric is commonly used to evaluate the quality and diversity of GAN samples?
- A) Accuracy
- B) mAP
- C) FID (Fréchet Inception Distance)
- D) BLEU

**10. (Day 4)** Inside a container, Flask logs `Running on http://127.0.0.1:5000`. With `docker run -p 5000:5000`, the host gets "connection reset". What is the fix?
- A) Use `-p 127.0.0.1:5000:5000`
- B) Add `EXPOSE 5000` to the Dockerfile
- C) Bind the server to `0.0.0.0`
- D) Use port 80 instead

**11. (Day 4)** In docker-compose, the Streamlit service should call the API service (named `api`) at:
- A) `http://localhost:5000`
- B) `http://127.0.0.1:5000`
- C) `http://api:5000`
- D) `http://host.docker.internal:8501`

**12. (Day 4)** A pod shows `Reason: OOMKilled, Exit Code: 137`. This means:
- A) The liveness probe failed
- B) The container exceeded its memory limit and was killed (SIGKILL)
- C) The image could not be pulled
- D) The CPU limit was exceeded

**13. (Day 4)** What happens when a container exceeds its **CPU** limit?
- A) It is OOM-killed
- B) It is evicted from the node
- C) It is throttled, so latency increases
- D) Nothing; CPU limits are advisory

**14. (Day 4)** `kubectl get hpa` shows `TARGETS <unknown>/70%`. Which pair of causes is most likely?
- A) Too many replicas; wrong image tag
- B) metrics-server not installed; no CPU **requests** set on the container
- C) The liveness probe is failing; the Service type is wrong
- D) The ConfigMap is missing; the Secret is not base64-encoded

**15. (Day 4)** Users see a few 502 or connection errors during every rolling update. Which combination best fixes this?
- A) Set `replicas: 1` and `maxSurge: 0`
- B) A `preStop` sleep of a few seconds, graceful shutdown on SIGTERM, and correct readiness probes
- C) Remove all probes
- D) Use the `:latest` tag

**16. (Day 4)** The first request to your Render free-tier API takes about 45 s, and later ones are fast. The main reason is:
- A) The model is not quantized
- B) The service spun down after inactivity, so the first request triggers a cold start (container start + model load)
- C) Render throttles HTTPS
- D) Flask caches responses after the first request

**17. (Day 5)** Which kind of drift can you detect directly **without** ground-truth labels?
- A) Concept drift (P(y|x) changes)
- B) Covariate / data drift (P(x) changes)
- C) Label noise
- D) None; all drift needs labels

---

## Section B — True / False

**18. (Day 1)** `torch.no_grad()` turns off Dropout and makes BatchNorm use running statistics.

**19. (Day 2)** If you set `requires_grad=False` on every backbone parameter, BatchNorm running mean and variance can no longer change, even while the model is in `train()` mode.

**20. (Day 2)** Pruning 50% of weights with unstructured magnitude pruning roughly halves inference latency when using standard dense CPU kernels.

**21. (Day 2 demo)** A steadily rising generator loss during GAN training proves that the generated images are getting worse.

**22. (Day 4)** In docker-compose, `depends_on: [api]` guarantees that the API is ready to serve requests before the UI container starts.

**23. (Day 4)** Rebuilding and pushing `myapi:latest`, then running `kubectl apply` on an unchanged manifest that references `myapi:latest`, triggers a rolling update to the new image.

**24. (Day 5)** Temperature scaling (dividing logits by a learned T > 0) improves calibration and also changes the model's top-1 accuracy.

---

## Section C — Short answer (2 points each)

**25. (Day 2)** MobileNetV2 has about 3.5 million parameters. Estimate its weight size in FP32 and in INT8, and give one reason the real INT8 file is not exactly 4× smaller.

**26. (Day 2)** Why do we fuse Conv + BatchNorm (+ ReLU) before quantizing a CNN? Give two reasons.

**27. (Day 4)** Give two reasons why a PyTorch-based serving image can be about 2 GB while an ONNX Runtime-based image is about 300 MB. Why does that difference matter in Kubernetes?

**28. (Day 4)** Explain the difference between a readiness probe and a liveness probe: what does Kubernetes do when each one fails? Why can pointing both at a "model loaded" check cause problems?

**29. (Day 4 / Fermi)** Your API must serve 1,000,000 predictions per day. Roughly how many requests per second is that on average? If peak is 4× average and each 1-vCPU pod handles about 26 req/s at safe utilization, how many pods do you need at peak?

**30. (Day 5)** Your model scored 95% in the notebook, but users say it is "often wrong". List **four** distinct possible causes, spanning at least two different layers (data, model, serving, infrastructure, users).

---
---

# Answer Key

### Section A

| Q | Answer | Explanation |
|---|---|---|
| 1 | **C** | Uniform over 10 classes: −ln(1/10) = ln 10 ≈ 2.303. (B is ln 2, for binary.) |
| 2 | **C** | 3×3×1×32 weights = 288, plus 32 biases = 320. |
| 3 | **B** | Upscaling adds no information. It matches pretrained scale statistics. MobileNetV2's stride of 32 would turn a 32 px input into a 1×1 map, while 96 px gives 3×3. It *increases* compute (9× vs 32 px). |
| 4 | **B** | Default dynamic quantization covers Linear and recurrent layers. MobileNetV2 has only one small Linear layer. Use static PTQ or QAT for CNNs. |
| 5 | **B** | Weights are known offline. Activation ranges depend on the data, so representative samples are run through observers to set scales and zero-points. |
| 6 | **C** | Compression tends to "forget" the long tail first. Always compare per-class and per-slice metrics before and after. |
| 7 | **B** | Class 0 = background. With `num_classes=2` for background + person, pedestrians must be labelled 1. |
| 8 | **C** | The second stage runs once per proposal, so fewer proposals means linearly less RoI-head work, at a risk to recall in crowded scenes. |
| 9 | **C** | FID compares feature statistics of real vs generated images. Lower is better. It is only comparable under the same protocol and sample count. |
| 10 | **C** | The container's 127.0.0.1 is its own loopback. `EXPOSE` only documents the port. Bind to 0.0.0.0. |
| 11 | **C** | Compose provides DNS by service name. `localhost` inside the UI container is the UI container itself. |
| 12 | **B** | 137 = 128 + 9 (SIGKILL) from the OOM killer when the memory limit was exceeded. |
| 13 | **C** | CPU is compressible, so the process gets throttled (CFS quota). Memory is not, so exceeding it means OOMKilled. |
| 14 | **B** | The HPA needs metrics (metrics-server) and computes utilization as usage ÷ request. Without requests the percentage is undefined. |
| 15 | **B** | Endpoint removal and SIGTERM happen at the same time. A preStop sleep lets routing catch up, graceful shutdown finishes in-flight requests, and readiness keeps traffic away from pods that aren't ready. |
| 16 | **B** | Free instances spin down when idle (about 15 min on Render). A cold start includes the image, the Python start and the model load. Slim images and client timeouts or retries help. |
| 17 | **B** | Input distributions (and prediction or confidence distributions) can be monitored without labels. Concept drift generally needs labels. |

### Section B

| Q | Answer | Explanation |
|---|---|---|
| 18 | **False** | `no_grad` only turns off gradient tracking. `model.eval()` changes Dropout and BN behaviour. You need both at inference. |
| 19 | **False** | Running stats are buffers updated in the forward pass in train mode, independent of `requires_grad`. Put the BN layers in `eval()` to freeze them. |
| 20 | **False** | Zeros are still stored densely and multiplied. Real speed-ups need structured pruning, very high sparsity with sparse kernels, or hardware support (e.g. 2:4 sparsity). |
| 21 | **False** | GAN losses are relative to a discriminator that keeps changing. A rising G loss may just mean D improved. Judge by samples and FID. |
| 22 | **False** | `depends_on` only orders container start. Use a `healthcheck` with `condition: service_healthy`, plus client retries. |
| 23 | **False** | The pod template didn't change, so no rollout happens. Use immutable tags (git SHA or version) or `kubectl rollout restart`. |
| 24 | **False** | Dividing all logits by the same T > 0 keeps their order, so the argmax (and accuracy) is unchanged. Only the confidences change. |

### Section C (marking guide: 2 = complete and correct, 1 = partial)

**25.** FP32: 3.5M × 4 B ≈ **14 MB**. INT8: 3.5M × 1 B ≈ **3.5 MB**. Not exactly 4× because of: scale and zero-point metadata (per tensor or per channel); biases kept in int32 or fp32; some layers left in fp32 (first or last, unsupported ops); file-format overhead. Any one reason counts.

**26.** Any two of these:
- At inference, BN is a fixed affine transform that can be **folded into the conv weights and bias**, so there are fewer ops and memory passes.
- It **avoids rounding between conv and BN**: one quantized op instead of several requantization steps, which is better accuracy.
- Fused kernels (ConvReLU) are **faster**.
- Quantized backends *expect* fused patterns.

**27.** Reasons:
- The default Linux torch wheels include **CUDA/cuDNN libraries (GBs)**, and even CPU torch is hundreds of MB.
- torchvision and other extras.
- ORT is a lightweight inference runtime (tens of MB) on a slim base.

Why it matters in Kubernetes: image **pull time**, so slower pod start-up, slower autoscaling and longer cold starts; also node disk and registry storage and cost.

**28.**
- **Readiness** failure: the pod is **removed from Service endpoints**, it gets no traffic, and it is *not* restarted.
- **Liveness** failure: the container is **restarted**.
- If liveness checks "model loaded", a slow model load (e.g. 20 s under CPU limits) makes liveness fail and the pod is killed before it finishes, which becomes **CrashLoopBackOff**. Transient issues can also cause restart storms.
- Correct setup: cheap liveness, readiness = model loaded, and a **startupProbe** for slow start-up.

**29.** 10⁶ / 86,400 ≈ **11.6 req/s** average. Peak ≈ 4 × 11.6 ≈ 46 req/s. 46 / 26 ≈ 1.8, so **2 pods** at peak, and **3 recommended** for redundancy and rolling updates. Accept 2–3 with reasoning.

**30.** Any four, from at least two layers:
- **Data**: drift (different cameras, lighting), open-set inputs outside the training classes, test-set leakage or duplicates inflating the 95%, class imbalance hiding poor recall on important classes, subgroup failures.
- **Model**: poor calibration (confidently wrong), threshold chosen badly, INT8 variant degraded.
- **Serving**: preprocessing mismatch (RGB/BGR, mean/std, resize), `eval()` forgotten, wrong model version deployed, EXIF rotation.
- **Infrastructure**: timeouts, cold starts, errors perceived as "wrong".
- **Users**: expectations (want top-3), and only failures get reported (selection bias).

---

### Score bands (for feedback)

| Score (of 36) | Interpretation |
|---|---|
| 31–36 | Ready to own an ML service end-to-end |
| 24–30 | Solid; review the missed topics in `question_bank.md` |
| 16–23 | Revisit the Day 2 optimization and Day 4 Kubernetes material |
| < 16 | Pair up with a stronger peer on the Day 3–4 project; retake in a week |
