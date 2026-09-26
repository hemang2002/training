# Day 5 Question Bank — Thinking Like an ML Engineer

**106 questions** in 10 topics. Each one has a difficulty, the key points you should expect in an answer, and follow-ups to push further.

| Difficulty | Meaning | Typical use |
|-----------|---------|-------------|
| ★ | Recall plus one step of reasoning | Warm-up rapid-fire (30–45 s) |
| ★★ | Needs a causal explanation or a trade-off | "Why?" deep dives, cold calls |
| ★★★ | Open-ended, multi-factor, or needs quantitative reasoning | Think-pair-share, golden questions |

**How to use this bank:** ask the question, wait, take an answer, then use the follow-ups. The model answers are what a *strong* answer covers; partial answers still earn credit. Do not read the model answer aloud before the students have tried.

## Contents

1. [PyTorch & Training](#1-pytorch--training) (Q1.1–Q1.10)
2. [Transfer Learning](#2-transfer-learning) (Q2.1–Q2.10)
3. [Optimization: Quantization, Pruning, Export](#3-optimization-quantization-pruning-export) (Q3.1–Q3.11)
4. [Detection & GANs](#4-detection--gans) (Q4.1–Q4.11)
5. [APIs & Serving](#5-apis--serving) (Q5.1–Q5.11)
6. [Docker](#6-docker) (Q6.1–Q6.10)
7. [Kubernetes](#7-kubernetes) (Q7.1–Q7.11)
8. [Cloud & Cost](#8-cloud--cost) (Q8.1–Q8.10)
9. [MLOps, Monitoring & Data Drift](#9-mlops-monitoring--data-drift) (Q9.1–Q9.11)
10. [Ethics & Responsible AI](#10-ethics--responsible-ai) (Q10.1–Q10.11)

---

## 1. PyTorch & Training

### Q1.1 ★ — Why do we call `optimizer.zero_grad()` every step? What happens if we forget?
**Key points**
- `.backward()` **adds** into `.grad`; it does not overwrite. This is on purpose, so you can accumulate gradients across several backward passes or losses.
- If you forget, step *t* uses the sum of all gradients so far. The effective step size keeps growing, updates point in stale directions, and training usually diverges or oscillates.

**Follow-ups**
- How would you use this behaviour on purpose to simulate batch size 256 on a GPU that only fits 32? *(Gradient accumulation: divide the loss by 8, call `backward()` 8 times, then `step()` and `zero_grad()` once.)*
- Does gradient accumulation give exactly the same result as a real batch of 256 when BatchNorm is used? *(No. BN statistics are still computed per micro-batch of 32.)*

### Q1.2 ★ — For a 10-class classifier (FashionMNIST, CIFAR-10) with random init, what loss should you see at step 0? Why is that useful?
**Key points**
- With near-uniform predictions, cross-entropy is −ln(1/10) = **ln 10 ≈ 2.303**.
- This is a free sanity check. A loss of 15 means logits are badly scaled or labels are wrong. A loss near 0 before any training means label leakage or a bug in how the loss is computed.

**Follow-ups**
- What is the expected initial loss for binary cross-entropy? *(ln 2 ≈ 0.693.)*
- Your accuracy is stuck at exactly 10% and the loss is stuck at 2.303. What does that tell you? *(The model outputs a constant. Suspect a dead network, too high an LR, or the gradient not flowing.)*

### Q1.3 ★★ — `model.eval()` and `torch.no_grad()` both appear at inference time. Are they redundant?
**Key points**
- They do different jobs. `eval()` changes **layer behaviour**: Dropout is turned off and BatchNorm uses its running statistics instead of batch statistics.
- `no_grad()` (or `inference_mode()`) turns off **graph recording**. That saves memory and time but does not change what the layers compute.
- You need both. Forgetting `eval()` gives wrong, batch-dependent outputs. Forgetting `no_grad()` gives correct outputs but wastes memory.

**Follow-ups**
- Can you compute gradients with the model in eval mode? *(Yes. Adversarial attacks and saliency maps do this.)*
- What happens if you *train* with the model accidentally left in eval mode? *(BN stats never update and dropout is off, so you get more overfitting and stale normalization.)*

### Q1.4 ★★ — Why does BatchNorm behave badly with batch size 2, and why does torchvision's Faster R-CNN use `FrozenBatchNorm2d`?
**Key points**
- BN normalizes with the batch mean and variance. With 2 samples those estimates are very noisy, so training becomes noisy and the running statistics come out poor.
- Detection trains on few large images per batch (often 2), so torchvision freezes the BN statistics and affine parameters from pretraining.
- Alternatives: GroupNorm or LayerNorm (independent of the batch), synchronized BN across GPUs, or freezing BN.

**Follow-ups**
- What does BN do with batch size 1 in train mode on a 1×1 feature map? *(It raises an error: it needs more than one value per channel.)*
- Why don't Transformers use BatchNorm? *(Variable sequence lengths, small batches, and LayerNorm works independently per token.)*

### Q1.5 ★★ — Why normalize inputs with mean and std? What happens if training normalizes but the API forgets?
**Key points**
- Normalized inputs make optimization better conditioned: similar scales mean the loss surface is less elongated and gradients are better behaved. Pretrained weights also *expect* that exact input distribution.
- If the API forgets, the inputs are shifted and scaled (for example [0, 1] instead of about [−2, 2.6]). **Nothing crashes.** Predictions just degrade quietly, often collapsing to one or two classes.

**Follow-ups**
- How would you catch this before users do? *(A golden test: a fixed image goes through the training pipeline and the API, and the logits are compared within a tolerance.)*
- Which is worse, a missing `/255` or a missing mean/std step? Why?

### Q1.6 ★★ — Validation loss starts *rising* after epoch 5 while validation accuracy keeps *improving*. How is that possible?
**Key points**
- Accuracy only looks at the argmax. Cross-entropy also penalizes confidence.
- The model gets more examples right, but becomes extremely overconfident on the few it gets wrong. Each of those costs a huge −log p, so the mean loss rises.
- This is a sign of overconfidence and poor calibration.

**Follow-ups**
- Which checkpoint would you ship, and which metric decides? *(The metric that matches the product. If you threshold on confidence, loss and calibration matter.)*
- How would you fix the confidence without retraining? *(Temperature scaling. See Q9.4.)*

### Q1.7 ★★ — Adam vs SGD with momentum: why might you pick each?
**Key points**
- Adam adapts a step size per parameter. It converges fast, is robust to the LR choice, and is the default for Transformers and quick experiments.
- SGD with momentum and a good LR schedule often **generalizes slightly better** on CNNs for vision (the classic ImageNet recipes), and it uses less memory.
- AdamW decouples weight decay from the gradient-scaled update. Plain Adam with L2 is not true weight decay.
- Memory: Adam keeps 2 extra fp32 tensors per parameter (m and v).

**Follow-ups**
- How much memory do the optimizer states need for ResNet18 (11.7M params) under Adam? *(2 × 11.7M × 4 B ≈ 94 MB.)*
- Why is L2 regularization inside Adam not the same as weight decay? *(The L2 gradient gets divided by √v, so parameters with large gradients are decayed less.)*

### Q1.8 ★★ — Why does a CNN need far fewer parameters than an MLP on images?
**Key points**
- **Weight sharing and locality**: the same small kernel slides over every position.
- Example: 784→512 dense = 401,920 params. A 3×3 conv with 32 filters on 1 channel = 3·3·1·32 + 32 = **320** params.
- This builds in a useful bias: convolution is translation **equivariant**, and pooling adds approximate invariance.

**Follow-ups**
- Is a CNN rotation invariant? *(No. You need augmentation or special architectures.)*
- Fewer parameters does *not* mean fewer FLOPs. Why? *(Conv FLOPs scale with H×W. Each weight is reused at every spatial position.)*

### Q1.9 ★★★ — Why does training need so much more memory than inference, and why does doubling the image side roughly quadruple it?
**Key points**
- Backprop needs the **stored activations** of every layer to compute gradients. Inference can throw each activation away once the next layer has used it.
- Training memory ≈ weights + gradients + optimizer states + **activations × batch size**. Activations usually dominate for CNNs.
- Activation size scales with H×W. Doubling the side gives 4× the pixels, so about 4× the activation memory and compute.
- Mitigations: smaller batch, mixed precision, gradient checkpointing (recompute instead of storing), lower resolution.

**Follow-ups**
- Why did we train on CIFAR-10 at 96 px and not 224 px? *((224/96)² ≈ 5.4× more activation memory and compute.)*
- What does gradient checkpointing trade? *(About 20–30% extra compute for large activation savings.)*

### Q1.10 ★★★ — You set `torch.manual_seed(42)` and two runs still give 91.2% vs 90.6%. Why? And can you now claim your new trick's +0.5% gain?
**Key points**
- Sources of nondeterminism: cuDNN algorithm selection and nondeterministic kernels (atomic adds), DataLoader worker seeding, different GPU or CPU, library versions, and floating-point non-associativity in parallel reductions.
- `torch.use_deterministic_algorithms(True)` plus `cudnn.benchmark=False` plus seeded workers help, but can be slower.
- **The key insight:** if the noise between runs is about ±0.5%, a +0.5% gain from one run proves nothing. Run 3–5 seeds and report mean ± std or confidence intervals.

**Follow-ups**
- How many seeds would convince you? What statistical test would you use?
- Is nondeterminism a problem in production inference? *(Usually tiny numeric differences. It matters for audit and regulated settings.)*

---

## 2. Transfer Learning

### Q2.1 ★ — MobileNetV2 was trained on ImageNet (dogs, cars…). Why does it help on CIFAR-10 or on a totally new task?
**Key points**
- Early layers learn **generic features** (edges, colour blobs, textures). Middle layers learn parts. Only late layers are task-specific.
- Pretrained weights are a far better starting point than random ones: faster convergence, much less data needed, better generalization.

**Follow-ups**
- When might ImageNet pretraining help little? *(Very different modalities: spectrograms, multispectral satellite imagery with more than 3 channels, some medical imaging. Even then it often still helps a bit.)*
- How would you check *which* layers transfer well? *(Linear probes on features from different layers.)*

### Q2.2 ★★ — Why did we upscale CIFAR-10 from 32×32 to 96×96? Upscaling adds no information, so why does it help?
**Key points**
- MobileNetV2 downsamples by 32 overall. A 32×32 input ends as a **1×1 feature map**, so spatial structure is destroyed early, and objects are at a far smaller scale than the filters learned on 224 px ImageNet.
- Upscaling adds no information, but it matches the **scale statistics and receptive fields** the pretrained filters expect. At 96 px the final feature map is 3×3.
- The cost: compute scales with pixels, so (96/32)² = **9×** the FLOPs of 32 px. 224 px would be another 5.4× on top.

**Follow-ups**
- How would you choose between 64, 96, 128 and 224 px? *(Plot accuracy against latency or cost and pick the knee of the curve.)*
- Could you instead change the network's first stride? What would you lose? *(You could, but the pretrained filters at later stages then see a different scale.)*

### Q2.3 ★★ — Feature extraction (frozen backbone) vs full fine-tuning: how do you decide?
**Key points**

| Data size \ Domain similarity | Similar | Different |
|---|---|---|
| Small | Freeze backbone, train head | Freeze early layers, fine-tune later ones carefully |
| Large | Fine-tune everything (low LR) | Fine-tune everything; consider training longer |

- Use **smaller LRs for pretrained layers** (discriminative LRs) so you don't destroy useful features (catastrophic forgetting).
- A common recipe: train the head first for a few epochs, then unfreeze and fine-tune at a low LR.

**Follow-ups**
- Why train the head first before unfreezing? *(A random head sends large, noisy gradients into the backbone.)*
- How does freezing affect training memory and speed? *(No gradients or optimizer states for frozen layers, and no backward pass through them.)*

### Q2.4 ★★ — You freeze the backbone (`requires_grad=False`) but call `model.train()`. Is the backbone really frozen?
**Key points**
- **No.** `requires_grad=False` freezes BN's γ and β, but BN's **running mean and variance still update** in train mode, because they are buffers updated in the forward pass, not by the optimizer.
- The backbone's behaviour drifts toward the new dataset's statistics. Sometimes that helps. Often it causes a quiet train/eval mismatch.
- To truly freeze, put the BN modules in eval mode after calling `model.train()`.

**Follow-ups**
- Why can this be worse with small batch sizes?
- How would you write a unit test proving the backbone didn't change? *(Hash or compare the `state_dict` before and after one training step.)*

### Q2.5 ★★ — Why MobileNetV2 and not ResNet50 for this course?

**Key points**

| Model | Params | MACs @224 | ImageNet top-1 | fp32 size |
|---|---|---|---|---|
| MobileNetV2 | ≈3.5M | ≈0.3 G | ≈72% | ≈14 MB |
| ResNet18 | ≈11.7M | ≈1.8 G | ≈70% | ≈47 MB |
| ResNet50 | ≈25.6M | ≈4.1 G | ≈76% | ≈102 MB |

- MobileNetV2 was designed for mobile and CPU deployment (depthwise separable convolutions, inverted residuals). It is a good fit for CPU serving, free tiers and edge devices.

**Follow-ups**
- MobileNetV2 has about 13× fewer MACs than ResNet50. Is it 13× faster on a GPU? *(No. Depthwise convs have low arithmetic intensity and are memory-bound, so GPUs are underused.)*
- When would you pick ResNet50 anyway? *(When accuracy matters most, you have GPU serving, or you need robustness to quantization.)*

### Q2.6 ★★★ — Derive how much a depthwise separable convolution saves compared with a standard convolution.
**Key points**
- Standard conv cost: k²·C_in·C_out·H·W.
- Depthwise (k²·C_in·H·W) plus pointwise 1×1 (C_in·C_out·H·W).
- Ratio = 1/C_out + 1/k². For k = 3 and a large C_out that is about 1/9, so **8–9× fewer** FLOPs and parameters.
- MobileNetV2 adds **inverted residuals** (expand → depthwise → project) and **linear bottlenecks** (no ReLU on the narrow projection).

**Follow-ups**
- Why no ReLU after the narrow bottleneck? *(ReLU in a low-dimensional space destroys information by zeroing it.)*
- Why are depthwise layers the hardest part of MobileNet to quantize? *(Weight ranges differ widely per channel, so per-tensor scales fit badly. Per-channel quantization fixes most of it.)*

### Q2.7 ★★ — A linear probe (frozen features + linear head) gets 85% and full fine-tuning gets 95%. What does the gap tell you?
**Key points**
- Pretrained features are useful but **not perfectly aligned** with the new task. The domain or scale differs (for example low-resolution upscaled CIFAR vs ImageNet).
- A big gap argues for full or partial fine-tuning and says there is signal in adapting mid-level features.
- A small gap means feature extraction is enough: cheaper, less overfitting risk, and one shared backbone can serve many heads.

**Follow-ups**
- If you must serve 5 different classifiers on the same images, which approach makes deployment cheaper? *(A shared frozen backbone with 5 heads, so the backbone runs once.)*
- How would you check whether the 95% is overfitting the validation set through repeated tuning?

### Q2.8 ★★ — You have only 200 labelled images for a new product-defect class. What are your options?
**Key points**
- Transfer learning with a frozen backbone, strong augmentation, and class-balanced sampling or loss.
- Few-shot or metric learning (embeddings + nearest neighbour), or anomaly detection trained on "good" parts only.
- Get more data: active learning (label the most uncertain images), synthetic data, weak labels.
- Evaluate carefully: stratified k-fold, and report confidence intervals.

**Follow-ups**
- With 200 images, how wide is the 95% confidence interval on a measured 90% accuracy from a 40-image test set? *(Roughly ±9 percentage points, from √(0.9·0.1/40) ≈ 0.047 × 1.96.)*
- Why can anomaly detection be the better framing for defects? *(Defects are rare and varied. "Normal" is easy to collect.)*

### Q2.9 ★★ — Should augmentation be applied at test time? Which augmentations are *dangerous* for some tasks?
**Key points**
- Standard evaluation uses deterministic transforms (resize, center-crop, normalize).
- Test-time augmentation (TTA) averages predictions over flips or crops. It often gives +0.5–1%, but costs N× latency.
- Augmentations can break the label: horizontal flips for text, digits or X-ray laterality ("left" lung); rotation for "6 vs 9"; colour jitter when colour *is* the defect or the diagnosis; random crops that cut out a small object.

**Follow-ups**
- Would you use TTA in the Day 3 API? What does it do to cost per prediction?
- What augmentation would make a pedestrian detector more robust at night? *(Brightness/gamma changes, noise, blur. Real night data is still better.)*

### Q2.10 ★★ — You fine-tuned on CIFAR-10 at 96 px. The deployed app receives 12-megapixel phone photos. What can go wrong?
**Key points**
- The domain gap is huge: CIFAR images are tiny, centred and low-quality, while phone photos are high-resolution, cluttered and have different framing. Downscaling a 4000×3000 image to 96×96 gives a very different image from an upscaled 32×32 CIFAR image (different blur and aliasing).
- EXIF orientation (the photo arrives rotated), aspect-ratio distortion if you resize without cropping, JPEG artefacts.
- Classes outside the 10 CIFAR classes still get a confident prediction (the open-set problem).

**Follow-ups**
- How would you measure this gap *before* launch? *(Collect a small labelled set of real phone photos. That is your real test set.)*
- Which preprocessing choices (resize, then center-crop, then normalize) must exactly match training?

---

## 3. Optimization: Quantization, Pruning, Export

### Q3.1 ★★ — Why does INT8 quantization usually cost only about 0.5–2% accuracy even though we throw away 24 of 32 bits?
**Key points**
- Trained weights and activations sit in **narrow, roughly bell-shaped ranges**. 256 levels with a well-chosen scale (per channel for weights) give a small *relative* rounding error.
- Networks are **robust to small noise**: training with SGD noise, dropout and augmentation pushes them toward flat minima where small perturbations barely change the output.
- Accumulation happens in **int32**, so the dot products are not truncated. Only the stored values are coarse.
- Classification only needs the **argmax** to stay the same, and small logit noise rarely flips it.
- Rounding errors are roughly independent and partly **average out** across large dot products.

**Follow-ups**
- Why is MobileNetV2 more sensitive than ResNet50? *(Depthwise layers have fewer weights to average errors over and very different per-channel ranges. Its compact design has less redundancy.)*
- Why do people often keep the first and last layers in fp32? *(The first layer sees raw input with few channels. The last layer directly produces logits, so errors there are not averaged away.)*

### Q3.2 ★★ — Why doesn't 50% unstructured pruning make the model faster, or even smaller on disk?
**Key points**
- Unstructured pruning sets individual weights to zero, but tensors stay **dense**. Dense matmul and conv kernels multiply the zeros anyway, so the FLOPs executed are the same.
- Sparse formats (CSR and similar) add index overhead and irregular memory access. They usually only win above roughly **90% sparsity** or with hardware support (NVIDIA Ampere 2:4 structured sparsity).
- `torch.nn.utils.prune` adds a `weight_orig` parameter and a `weight_mask` buffer, so the model gets **bigger** until `prune.remove()`.
- The file is only smaller after compression (zip/gzip compresses zeros well) or sparse storage.
- **Structured pruning** (removing whole channels or filters) shrinks tensor shapes and does give real speed-ups.

**Follow-ups**
- What is 2:4 sparsity and why can hardware speed it up? *(In each group of 4 weights, 2 are zero. The pattern is regular, so hardware can skip them efficiently.)*
- If structured pruning is faster, why doesn't everyone use it? *(It costs more accuracy for the same sparsity and needs fine-tuning.)*

### Q3.3 ★★ — We applied PyTorch *dynamic* quantization to MobileNetV2 and saw almost no speed-up or size reduction. Why?
**Key points**
- PyTorch dynamic quantization (`quantize_dynamic`) only targets `nn.Linear` and `nn.LSTM`/`GRU` by default, **not `Conv2d`**.
- MobileNetV2's compute and most of its parameters are in convolutions. The classifier is a single Linear(1280→10), so a tiny fraction gets quantized.
- For CNNs use **static PTQ** (calibrated activation ranges, fused conv+BN+ReLU) or **QAT**, or ONNX Runtime static quantization.

**Follow-ups**
- Which model families benefit most from dynamic quantization? *(Transformers like BERT, and LSTMs, where Linear layers dominate compute.)*
- Why is dynamic quantization called "dynamic"? *(Activation scales are computed at runtime per batch. Only weights are pre-quantized.)*

### Q3.4 ★★ — Static PTQ needs a calibration dataset. Why? How big should it be, and what happens if it is unrepresentative?
**Key points**
- Weight ranges are known offline. **Activation ranges depend on the inputs**, so we run representative data through observers to choose each activation's scale and zero-point.
- Usually **100–1000 samples** are enough. More helps little. Representativeness matters more than count.
- Calibrate on noise, on one class, or on differently preprocessed images, and the ranges are wrong: activations get clipped or quantized coarsely, and accuracy can collapse.

**Follow-ups**
- MinMax vs histogram/entropy observers: which is more robust to outliers? *(Histogram or percentile, because MinMax lets one outlier stretch the range.)*
- Should the calibration set come from the train set or the validation set? *(Train, or held-out data that is not the test set, to avoid leakage into the reported numbers.)*

### Q3.5 ★★ — When is QAT worth its extra cost over PTQ?
**Key points**
- When PTQ loses too much accuracy (more than 1–2%), typical for compact models (MobileNets) or low bit-widths (4-bit).
- QAT inserts **fake-quantize** ops during training, so the weights adapt to the rounding. Gradients pass through the rounding with the **straight-through estimator** (STE).
- Cost: extra fine-tuning epochs, more complex code, and the need for the training data.

**Follow-ups**
- Why is the STE needed? *(Rounding has zero gradient almost everywhere. The STE treats it as identity in the backward pass.)*
- Why freeze BN statistics and observers near the end of QAT?

### Q3.6 ★★ — FP16 halves the model size. Why might FP16 inference be *slower* than FP32 on a CPU?
**Key points**
- Many CPUs have no fast FP16 arithmetic path. Values are converted to fp32 for compute, which adds overhead and gains nothing.
- GPUs with Tensor Cores (and some newer CPUs and ARM chips) have native FP16/BF16 units and get big speed-ups.
- The size halves either way, which still helps downloads and memory.

**Follow-ups**
- Why is bfloat16 preferred for *training*? *(It has fp32's 8-bit exponent range, so there is no overflow or underflow and no loss scaling is needed.)*
- Which is better for CPU inference, FP16 or INT8? *(INT8, which has good CPU kernel support, for example VNNI on x86 and dot-product instructions on ARM.)*

### Q3.7 ★★★ — Your static-quantized INT8 model is **slower** than FP32. List the possible reasons.
**Key points**
- **Wrong backend or engine** for the CPU (`fbgemm`/`x86` for x86, `qnnpack` for ARM). Quantized a model for x86 and ran it on an ARM box (Oracle Ampere, Raspberry Pi, M-series Mac).
- **Quant/DeQuant churn**: unsupported ops fall back to fp32, so the graph keeps converting back and forth.
- Modules **not fused** (conv+bn+relu), so there are extra passes and requantization steps.
- The CPU lacks fast int8 instructions (no AVX512-VNNI/AVX2 path).
- Model too small, or batch 1: fixed overheads dominate.
- Thread settings differ between the two benchmarks (fp32 used 8 threads, int8 used 1).
- A benchmarking error: no warm-up, timing the first call, including preprocessing or model loading.

**Follow-ups**
- What tool would you use to find the slow ops? *(`torch.profiler`, or ONNX Runtime's profiling output (`enable_profiling`).)*
- Write a fair benchmark protocol. *(Warm-up, fixed threads, the same input shape, N ≥ 100 runs, report median and p95, same machine, same power mode.)*

### Q3.8 ★★ — Why do we fuse Conv + BatchNorm (+ ReLU) before quantizing or exporting?
**Key points**
- At inference, BN is a fixed affine transform, so it can be **folded into the conv's weights and bias**. That is one op instead of two or three.
- Fewer memory passes, fewer kernel launches, and, for quantization, no intermediate rounding between conv and BN.

**Follow-ups**
- Why can't you fold BN during training? *(Training uses batch statistics, which change each step.)*
- Write the folding formula. *(W' = W·γ/√(σ²+ε), b' = (b−μ)·γ/√(σ²+ε) + β.)*

### Q3.9 ★★ — Why is ONNX Runtime often faster and *much* lighter to deploy than PyTorch eager?
**Key points**
- ORT works on a **static graph**: constant folding, operator fusion, memory planning, and optimized CPU kernels (MLAS) with execution providers (CPU, CUDA, TensorRT, OpenVINO…).
- No Python overhead per op.
- **A much smaller dependency**: the onnxruntime wheel is tens of MB, against hundreds of MB to GBs for torch. That means smaller Docker images and faster cold starts.

**Follow-ups**
- What does `torch.compile` offer instead, and why might you still export to ONNX? *(Compile speeds up training and inference inside Python/torch. ONNX gives portability and a light runtime.)*
- Name an op or pattern that commonly breaks ONNX export. *(Data-dependent control flow, some custom ops, dynamic shapes without `dynamic_axes`.)*

### Q3.10 ★★★ — Batching improves throughput but hurts latency. When would you batch inside an inference API?
**Key points**
- Batching amortizes per-call overhead and uses SIMD/GPU parallelism better. The gain is huge on GPUs and modest on CPUs.
- Requests must **wait** to form a batch, so latency rises. **Dynamic batching** (as in Triton or TorchServe) uses a max batch size and a max wait time (such as 5 ms).
- Worth it at high QPS or on a GPU. Not worth it at low QPS on a CPU with a small model.

**Follow-ups**
- Why do we care about p99 latency rather than the mean? *(Tail latency is what users notice. One slow call in a fan-out of many calls slows the whole page.)*
- With 11.6 req/s average, would you bother with dynamic batching? *(Probably not on a CPU.)*

### Q3.11 ★★ — Knowledge distillation: why does a small "student" trained on a big "teacher's" soft outputs beat the same student trained on hard labels?
**Key points**
- Soft targets (temperature-scaled softmax) carry **"dark knowledge"**: which wrong classes are similar ("shirt looks like T-shirt, not sneaker"). That is a richer signal per example.
- It acts as label smoothing and regularization, and the teacher can label unlabelled data too.
- Loss = α·CE(hard) + (1−α)·T²·KL(soft_teacher ‖ soft_student).

**Follow-ups**
- Why multiply by T²? *(Gradient magnitude from the soft targets scales as 1/T², so this keeps it balanced.)*
- Distillation vs quantization vs pruning: can you combine them? In what order? *(Usually distill to a smaller architecture, then quantize.)*

---

## 4. Detection & GANs

### Q4.1 ★★ — Lowering Faster R-CNN's input resolution speeds it up a lot. What do you lose, and how do you choose?
**Key points**
- Backbone and FPN compute scale roughly with pixel count. Halving the image side gives about 4× less backbone compute.
- **Small or distant objects disappear**: at low resolution a far pedestrian may be a few pixels tall, below what the anchors and features can resolve. Recall on small objects drops first.
- Choose by plotting mAP (or recall on small objects) against latency, *on your own data distribution*.

**Follow-ups**
- torchvision's `fasterrcnn_mobilenet_v3_large_320_fpn` vs `..._fpn` (min_size 800): when would you use each?
- Could you resize adaptively per frame? *(Yes: tiling, or crop regions of interest from a coarse first pass.)*

### Q4.2 ★★ — Why does reducing RPN proposals (for example 1000 → 100 at test time) speed up Faster R-CNN? When is it dangerous?
**Key points**
- The second stage runs RoIAlign and the box head **once per proposal**, so its cost grows linearly with the number of proposals.
- Penn-Fudan images usually hold only a few pedestrians, so 100 proposals is plenty.
- Dangerous for **crowded scenes** (a street crossing or stadium): true objects never get a proposal and recall drops.

**Follow-ups**
- What torchvision parameters control this? *(`rpn_pre_nms_top_n_test`, `rpn_post_nms_top_n_test`, `box_detections_per_img`.)*
- How would you set it for a crowd-counting product?

### Q4.3 ★★ — Two-stage (Faster R-CNN) vs one-stage (YOLO, SSD, RetinaNet): trade-offs?
**Key points**
- Two-stage: propose, then classify and refine each region. Historically more accurate, especially on small objects, but slower and more complex to export.
- One-stage: dense prediction in one pass. Faster, simpler to deploy, and modern versions are very accurate.
- Deployment angle: one-stage models are usually easier to export to ONNX/TensorRT and quantize, which is why edge deployments favour them.

**Follow-ups**
- For real-time on a drone (Case 4), which would you pick and why?
- Where do anchors come in, and what do anchor-free detectors change?

### Q4.4 ★★ — What does Non-Maximum Suppression do, and when does it hurt?
**Key points**
- It removes duplicate boxes: keep the highest-scoring box and suppress others of the same class with IoU above a threshold (such as 0.5).
- It fails in **crowds**: two overlapping pedestrians have high IoU, so one gets suppressed and you miss a person.
- Alternatives: Soft-NMS (decay scores instead of deleting), learned NMS, or NMS-free detectors.

**Follow-ups**
- Would you raise or lower the IoU threshold for crowd counting? What is the side effect? *(Raise it to keep overlapping people, at the cost of more duplicate boxes.)*
- Is NMS usually inside the exported ONNX graph? What does that do to portability?

### Q4.5 ★★ — Why is "accuracy" meaningless for detection? What do mAP@0.5 and mAP@[.5:.95] each tell you?
**Key points**
- Detection has no fixed number of predictions per image. You need matching (by IoU), precision and recall, and a confidence threshold.
- **AP** = area under the precision-recall curve for one class. **mAP** averages it over classes.
- mAP@0.5 is lenient on localization. mAP@[.5:.95] (the COCO metric) rewards tight boxes.

**Follow-ups**
- The product manager asks "how accurate is it?" How do you answer in business terms? *(Give precision and recall at the operating threshold: "Finds 92% of pedestrians; 1 false alarm per 20 frames.")*
- For a safety application, which matters more, precision or recall?

### Q4.6 ★★ — Penn-Fudan has only 170 images (345 pedestrians). Why is fine-tuning a detector on so little data feasible?
**Key points**
- The detector is pretrained on **COCO, which already includes "person"**. Backbone, FPN and RPN already know how to find people.
- We only replace the **box predictor head** (2 classes: background + person) and fine-tune briefly.
- The features are highly relevant, so the data needed is small.

**Follow-ups**
- What if the new class were "forklift" (not in COCO)? *(You would need more data, maybe more trainable layers, and would still get benefit from pretraining.)*
- With 170 images, how would you split train, val and test, and what does that do to confidence in your mAP?

### Q4.7 ★★ — Why are GANs notoriously hard to train?
**Key points**
- It is a **minimax game**, not a minimization. Each player's objective changes as the other learns, so there is no stable target.
- Failure modes: **mode collapse** (G produces a few outputs that fool D), **vanishing gradients** when D becomes too strong, oscillation instead of convergence.
- The losses don't track sample quality.
- DCGAN tricks: strided convs instead of pooling, BatchNorm, LeakyReLU in D, tanh output, Adam with lr = 2e-4 and β1 = 0.5.

**Follow-ups**
- Generator loss is going *up*. Is G getting worse? *(Not necessarily. D may be getting stronger. Look at samples and FID.)*
- How would you spot mode collapse quickly? *(Show a grid of samples from fixed noise vectors and look for low diversity. Measure diversity.)*

### Q4.8 ★★ — If GAN losses are meaningless, how do you evaluate a generator?
**Key points**
- **FID** (Fréchet Inception Distance): the distance between Gaussian fits of real and generated feature distributions. Lower is better.
- **Inception Score**, **precision/recall for generative models** (fidelity vs diversity), and human evaluation.
- Check **memorization**: nearest-neighbour search of generated samples against the training set.
- Caveat: FID uses ImageNet Inception features, which are not ideal for 28×28 grayscale FashionMNIST. Compare relative FIDs under the same protocol only.

**Follow-ups**
- Why is FID sensitive to sample count? *(It is a biased estimator. Always compare with the same N, such as 10k.)*
- Why do we check for memorization, and why is it a legal or ethical concern too?

### Q4.9 ★★ — For deployment, why do we export only the generator? What are its inputs and outputs?
**Key points**
- The discriminator is only a training-time critic, so it is not needed at inference.
- G's input is a latent vector z (for example shape [N, 100, 1, 1] for DCGAN). Its output is an image tensor in [−1, 1] from tanh, which must be rescaled to [0, 255] for display.
- Make the batch dimension dynamic in ONNX (`dynamic_axes`) so you can generate grids.

**Follow-ups**
- How would you make generation reproducible for a user ("regenerate my avatar")? *(Store the seed or the z vector.)*
- What does the API look like: input seed(s), output PNG or base64?

### Q4.10 ★★★ — Why does INT8 quantization of a GAN generator often hurt *visible* quality more than it hurts a classifier's accuracy?
**Key points**
- A classifier only needs the argmax to stay the same, so small logit errors are absorbed.
- A generator's output **is** the pixels. Every quantization error shows up as noise, banding or artefacts, with no argmax to absorb it.
- Transposed convolutions and the final tanh layer are sensitive. Output ranges are wide.
- Mitigate by keeping the last layer (and maybe the first) in fp32, using per-channel weights, QAT, or evaluating with FID rather than eyeballing.

**Follow-ups**
- How would you *measure* the quality loss from INT8 objectively? *(FID of int8 vs fp32 samples, or pixel/perceptual error for the same z.)*
- Is FP16 a better trade-off here? On which hardware?

### Q4.11 ★★ — Why is distilling a large generator into a small one easier than training a small GAN from scratch?
**Key points**
- Distillation turns an unstable adversarial game into **supervised regression**: for the same z, match the teacher's image (L1/L2 plus perceptual loss). The problem becomes stable and well-posed.
- The teacher gives unlimited paired training data (sample any z).
- You can optionally add a small adversarial loss for sharpness.

**Follow-ups**
- Why can pure L2 distillation give blurry outputs? *(L2 regresses to the mean. Perceptual or adversarial terms restore sharpness.)*
- How small can the student go before FID collapses? How would you find out?

---

## 5. APIs & Serving

### Q5.1 ★ — Why not run the Flask API with `app.run()` in production? How many gunicorn workers would you use?
**Key points**
- Flask's development server is not hardened or tuned for concurrency and warns you about this itself.
- Gunicorn (or uWSGI or waitress) manages multiple worker processes, restarts crashed workers, and handles timeouts.
- Worker count: the `2×cores+1` heuristic is for I/O-bound apps. For **CPU-bound inference**, use about `workers × intra-op threads ≈ available cores`.

**Follow-ups**
- Each gunicorn worker loads its own copy of the model. What does that do to memory? *(N × model memory. `--preload` can share read-only pages through copy-on-write, partly.)*
- What does gunicorn's `--timeout` protect against?

### Q5.2 ★★ — Where should the model be loaded: at startup or per request? What does that imply for health checks?
**Key points**
- At **startup, once**. Loading per request adds hundreds of milliseconds to seconds of latency and churns memory.
- Consequence: the container is **not ready** until loading finishes, so readiness must reflect "model loaded" (see Q7.2). Startup time also sets autoscaling and cold-start speed.
- Optionally run one **warm-up inference** at startup, because the first call is often slow (lazy initialization, memory allocation).

**Follow-ups**
- Lazy loading on the first request: when is it acceptable? *(Scale-to-zero demos. It hurts the first user.)*
- How would you hot-swap a new model version without restarting?

### Q5.3 ★★ — Why is preprocessing the number one source of *silent* serving bugs? How do you prevent it?
**Key points**
- Training uses torchvision transforms. The API often **reimplements** them with PIL, NumPy or OpenCV, and the mismatches never raise errors: RGB vs BGR (OpenCV loads BGR), HWC vs CHW, missing `/255`, wrong mean/std, resize interpolation (bilinear vs nearest, antialiasing), resize-then-crop vs direct resize, EXIF rotation, grayscale or RGBA inputs.
- Prevention: **one shared preprocessing module** used in training and serving (or baked into the ONNX graph), plus a **golden test** comparing logits for fixed images end-to-end.

**Follow-ups**
- Write the golden test in words. *(Save 5 images and their expected fp32 logits from the notebook. CI sends them through the API and asserts the difference is below 1e-3 and the top-1 matches.)*
- Should normalization live inside the ONNX model? Pros and cons? *(Pro: impossible to forget. Con: less flexibility, and the input contract must be documented.)*

### Q5.4 ★★ — Why should `/predict` return the model name, version and variant (fp32/int8) in its response?
**Key points**
- **Traceability**: when a user reports a wrong prediction, you know exactly which artefact produced it.
- It enables A/B tests, canaries, rollback verification, and debugging "it worked yesterday".

**Follow-ups**
- What else would you log per request? What must you *not* log? *(Log latency, variant, version, top-k, confidence, input shape or hash. Don't log raw images or PII without consent.)*
- How do you tie the version string to the exact training run? *(A git SHA plus a model-registry ID or file hash.)*

### Q5.5 ★★ — `/health` returns 200. Is the service healthy?
**Key points**
- Not necessarily. A **shallow** check (the process answers HTTP) is not the same as a **deep** check (model loaded, a dummy inference works, dependencies reachable).
- Different probes need different depth (see Q7.2).

**Follow-ups**
- Should the health endpoint run a full inference every 5 seconds? *(It costs CPU. Maybe run it on a cached schedule, or make it cheap.)*
- What should `/health` return while the model is still loading? *(503 for readiness.)*

### Q5.6 ★★ — Should `/predict` accept base64 in JSON or a multipart file upload?
**Key points**
- Base64 inflates the payload by **about 33%** and adds encode/decode CPU cost. It is simple for JSON clients.
- Multipart sends raw bytes and is more efficient for images.
- Either way: **limit upload size** (such as `MAX_CONTENT_LENGTH`), validate the image type, and handle decode errors.

**Follow-ups**
- Why is a size limit a security issue, not just a performance one? *(Memory exhaustion and denial of service, including decompression bombs.)*
- Could the client resize before uploading? What is the risk? *(Less bandwidth, but you now depend on the client's preprocessing matching yours.)*

### Q5.7 ★★ — 4 gunicorn workers × ONNX Runtime with 4 intra-op threads on a 4-core machine: latency is awful. Why?
**Key points**
- **Oversubscription**: 16 compute threads compete for 4 cores. Context switching, cache thrashing and lock contention follow.
- Fix: `workers × intra_op_num_threads ≈ cores` (for example 4 workers × 1 thread for throughput, or 1 worker × 4 threads for lowest single-request latency).
- In Kubernetes, ORT and PyTorch may see **all node cores**, not your CPU limit, so they spawn too many threads and get CFS-throttled.

**Follow-ups**
- Which configuration minimizes p50 latency at low load? Which maximizes throughput at high load?
- How do you set thread counts explicitly? *(`SessionOptions.intra_op_num_threads`, `torch.set_num_threads`, `OMP_NUM_THREADS`.)*

### Q5.8 ★★ — Why does the Streamlit UI call the API instead of loading the model itself?
**Key points**
- **Separation of concerns**: the UI and the inference service scale, deploy and fail independently. One model service can serve many clients (web, mobile, batch jobs).
- It gives one source of truth for model version and preprocessing.
- The cost: an extra network hop, one more service to run, and CORS or authentication to handle.

**Follow-ups**
- When is embedding the model directly in Streamlit perfectly fine? *(Single-user demos and quick prototypes.)*
- Streamlit reruns the whole script on each interaction. What does that mean if the model is loaded inside the script? *(Use `st.cache_resource`.)*

### Q5.9 ★★★ — You serve fp32 and int8 variants. How do you decide which one production should use, with evidence?
**Key points**
- Offline: accuracy (overall **and per class**), latency p50/p95, memory, and cost per 1k predictions on production-like hardware.
- Online: **shadow mode** (int8 runs on real traffic, responses discarded) and measure the **agreement rate** with fp32; then a canary with a small traffic share.
- Decide with an explicit threshold, such as "switch if agreement ≥ 99% and p95 latency improves by ≥ 30%".

**Follow-ups**
- Why is the agreement rate useful when you have no labels in production?
- Which classes would you expect to disagree most, and why? *(Classes that look alike, such as shirt vs T-shirt, and low-confidence cases.)*

### Q5.10 ★★ — What should the API do with a 50 MB PNG, a grayscale image, a CMYK JPEG, an animated GIF, or a text file named `cat.jpg`?
**Key points**
- Validate and normalize: size limit (return **413**), decode with error handling, `convert("RGB")` for grayscale, RGBA, palette or CMYK, take the first frame of a GIF, apply EXIF orientation.
- Status codes: **400/415/422** for client errors (bad or unsupported input), **500** for server bugs, **503** when the model is not ready or overloaded.
- Never let a bad input crash the worker.

**Follow-ups**
- Why does the 4xx vs 5xx distinction matter for monitoring and alerting? *(5xx means we are broken. 4xx means the client is. Alert on 5xx rate.)*
- What is a decompression bomb, and how does PIL guard against it? *(`Image.MAX_IMAGE_PIXELS`.)*

### Q5.11 ★★ — Would switching from Flask (sync) to FastAPI (async) speed up CPU-bound inference?
**Key points**
- **Not by itself.** Async helps when requests *wait* on I/O. CPU-bound inference inside an `async def` blocks the event loop and can make things worse.
- Throughput for CPU-bound work comes from processes and workers, thread tuning, batching, and faster models.
- FastAPI's real advantages are validation (pydantic), automatic OpenAPI docs, and handling many slow clients or I/O-heavy calls.

**Follow-ups**
- ONNX Runtime releases the GIL during `run()`. What does that allow? *(Threads can run inference in parallel inside one process.)*
- Where does async *genuinely* help in an ML service? *(Calling other services, streaming responses, fetching from object storage.)*

---

## 6. Docker

### Q6.1 ★★ — Why is the torch-based API image about 2 GB while the ONNX Runtime image is about 300 MB?
**Key points**
- The default `pip install torch` on Linux x86-64 pulls **CUDA-enabled** builds plus NVIDIA libraries (cuBLAS, cuDNN, NCCL…), which are **several GB installed**, even if you never use a GPU.
- Even the CPU-only torch wheel is several hundred MB installed, plus torchvision.
- The ORT image is python-slim (roughly 120–150 MB) plus onnxruntime (tens of MB), numpy, pillow and flask, so about 300 MB in total.
- Why it matters: **pull time → pod startup → autoscaling speed and cold starts**, plus registry storage, node disk and free-tier limits.

**Follow-ups**
- How do you install CPU-only torch? *(`pip install torch --index-url https://download.pytorch.org/whl/cpu`.)*
- At 50 MB/s, how long does a new node take to pull 2 GB vs 300 MB? *(About 40 s vs about 6 s, before extraction.)*

### Q6.2 ★ — Why does the order of Dockerfile instructions matter?
**Key points**
- Each instruction creates a **cached layer**. A change invalidates that layer and every layer after it.
- `COPY requirements.txt` → `pip install` → `COPY . .` means code edits don't reinstall dependencies.
- Use `pip install --no-cache-dir` and clean up inside the *same* `RUN` so the deleted files don't linger in an earlier layer.

**Follow-ups**
- Why doesn't `RUN rm -rf /tmp/big` in a *later* layer shrink the image? *(Earlier layers are immutable, so the file still ships.)*
- What should go in `.dockerignore` for an ML repo? *(Datasets, checkpoints, `.git`, `__pycache__`, notebooks, venv.)*

### Q6.3 ★★ — What does a multi-stage build remove? Why is Alpine often a *bad* base for Python ML?
**Key points**
- Multi-stage: build tools, compilers, headers and caches stay in the builder stage. The final stage copies only the artefacts (installed packages or a wheel directory, and the model).
- Alpine uses **musl libc**, and most scientific wheels (manylinux) target glibc. You end up compiling numpy or ORT from source, which is slow, fragile and often *bigger*.
- Prefer `python:3.x-slim` (Debian) or distroless.

**Follow-ups**
- What do you lose with distroless images? *(No shell, so debugging is harder. Use ephemeral debug containers.)*
- How would you measure which layer is big? *(`docker history`, `dive`.)*

### Q6.4 ★★ — Should model weights be baked into the image, downloaded at startup, or mounted as a volume?

**Key points**

| Option | Pros | Cons |
|---|---|---|
| Baked into image | Immutable, reproducible, works offline, simple | Bigger image; rebuild to change model; code and model versions coupled |
| Download at startup (S3/GCS/HF Hub) | Small image; model released independently | Startup depends on network and credentials; slower cold start; needs caching |
| Volume / PVC / init container | Shared across pods, decoupled | More infra; volume lifecycle; harder local dev |

- For small models (a MobileNetV2 ONNX file is about 3.5–14 MB), **baking in is usually best**.

**Follow-ups**
- How do you make sure the deployed image uses model v1.3 and not "whatever is in the bucket"? *(A pinned URI plus a checksum.)*
- What changes for a 7 GB model?

### Q6.5 ★★ — Flask listens on `127.0.0.1:5000` inside the container. `docker run -p 5000:5000` works, but the host gets "connection reset". Why?
**Key points**
- Each container has its **own network namespace**. Its 127.0.0.1 is the container's loopback, not the host's.
- Port forwarding delivers traffic to the container's external interface, where nothing is listening.
- Fix: bind to `0.0.0.0` (`app.run(host="0.0.0.0")` or `gunicorn -b 0.0.0.0:5000`).

**Follow-ups**
- What does `EXPOSE` actually do? *(It documents the port only. It does not publish it.)*
- Is binding to 0.0.0.0 a security issue on a VM? How do firewalls and security groups fit in?

### Q6.6 ★★ — In docker-compose, why must Streamlit use `http://api:5000` and not `http://localhost:5000`? And does `depends_on` guarantee the API is ready?
**Key points**
- Inside the Streamlit container, `localhost` means Streamlit's own container. Compose provides **DNS by service name** on the shared network.
- `depends_on` only orders **start-up**, not readiness. Use `depends_on: condition: service_healthy` with a `healthcheck`, and make clients retry.

**Follow-ups**
- How would you make the API URL configurable across local, compose and k8s? *(An environment variable such as `API_URL`.)*
- What is the Kubernetes equivalent of compose service-name DNS? *(A Service: `http://api-service:5000` or `api-service.namespace.svc.cluster.local`.)*

### Q6.7 ★★ — Why run the container as a non-root user? What typically breaks when you switch?
**Key points**
- Defence in depth: if the app is compromised (for example by a malicious upload exploiting an image decoder), root in the container makes container escape and host damage easier.
- Breakage: writing to root-owned paths (caches, `/app`), binding ports below 1024, pip caches under `/root`.
- Fix with `USER appuser` and `chown` in the Dockerfile. In k8s, `securityContext.runAsNonRoot`.

**Follow-ups**
- What is a read-only root filesystem, and what would the ML app need a writable `emptyDir` for? *(Temp files, HF or torch caches.)*
- Why scan images (Trivy, Grype)?

### Q6.8 ★★ — "Docker solves 'works on my machine'." What does it *not* solve?
**Key points**
- It does solve userland dependencies: Python, libraries, system packages.
- It does **not** solve:
  - the host **kernel**;
  - **CPU architecture** (an x86 image won't run natively on an ARM Oracle Ampere VM or an M-series Mac);
  - CPU instruction sets (AVX-512/VNNI, which affect int8 speed);
  - GPU drivers (the host must provide them);
  - memory and CPU differences;
  - external services, time zones, locale;
  - data.

**Follow-ups**
- How do you build an arm64 image on an x86 laptop? *(`docker buildx build --platform linux/arm64,linux/amd64`.)*
- Why can a container be fast on your laptop and slow on a cloud VM with the same vCPU count? *(vCPUs are often hyperthreads, other tenants share the host, CPUs are throttled, and cloud CPUs lack some instructions.)*

### Q6.9 ★★★ — The ONNX model file is 14 MB, but the container gets OOM-killed at a 512 MB limit. How?
**Key points**
- Memory is not just the model file:
  - the Python interpreter plus numpy, PIL and flask: tens of MB;
  - the ORT session: graph optimizations, kernels, **memory arena**;
  - **activations** per request multiplied by concurrent requests;
  - large decoded images (a 12 MP RGB image is 36 MB as uint8 and 144 MB as float32!);
  - **N gunicorn workers each holding a full copy**.
- Peak during loading can be about 2× (file bytes plus deserialized graph).

**Follow-ups**
- Which single change most reduces peak memory for large uploads? *(Downscale early, before converting to float, and cap the upload size.)*
- How do you measure container memory? *(`docker stats`, `kubectl top pod`, and the cgroup `memory.current`.)*

### Q6.10 ★ — Why pin exact versions (`onnxruntime==1.x.y`) in requirements.txt?
**Key points**
- Reproducibility. Unpinned builds silently change between builds.
- Real breakages: numpy 2.0 ABI changes, the ORT version not supporting the exported opset, Pillow changing default resampling behaviour.
- Use lock files (pip-tools, `uv lock`, poetry) and rebuild regularly to pick up security fixes *deliberately*.

**Follow-ups**
- What's the tension between pinning and security patches? How do tools like Dependabot help?
- Should you also pin the base image by digest?

---

## 7. Kubernetes

### Q7.1 ★★ — What happens to in-flight requests during a Kubernetes rolling update? How can users see errors?
**Key points**
- The Deployment creates new pods (up to `maxSurge`, default 25%). When they are **Ready**, it terminates old pods (keeping within `maxUnavailable`, default 25%).
- Terminating a pod: the pod is marked Terminating. **At the same time**: (a) the endpoint controller removes it from the Service's endpoints and kube-proxy or the ingress update their routing, and (b) the kubelet runs the `preStop` hook, then sends **SIGTERM**. After `terminationGracePeriodSeconds` (default 30 s) comes **SIGKILL**.
- **The race:** routing updates propagate asynchronously, so for a few seconds traffic can still arrive at a pod that is shutting down, giving connection refused or 502s. Requests still being processed are cut off if the app exits immediately on SIGTERM.
- Fixes: a `preStop` sleep of 5–10 s, graceful shutdown (gunicorn finishes in-flight requests on SIGTERM), a grace period longer than the slowest request, correct readiness probes, `maxUnavailable: 0`.

**Follow-ups**
- The new version has a bug but passes readiness. What happens, and how do you recover? *(The rollout completes. Use `kubectl rollout undo`, and consider canaries or progressive delivery.)*
- What do long-lived connections (websockets, Streamlit sessions) do during a rollout?

### Q7.2 ★★ — Why might readiness and liveness probes need *different* endpoints or logic?
**Key points**
- **Liveness** asks "is this process broken beyond repair?" Failure means **restart the container**.
- **Readiness** asks "should this pod receive traffic *right now*?" Failure means **remove it from Service endpoints**, with no restart.
- Using the same deep check for both causes trouble:
  - while the model loads for 20 s, liveness fails, the pod is killed, and it loops in CrashLoopBackOff forever;
  - if a downstream dependency (DB, feature store) blips, every pod fails liveness and restarts at once, a **cascading failure**;
  - under heavy load, slow responses fail liveness and restarts make the overload worse.
- Pattern: liveness is a cheap "process responds" check; readiness is "model loaded and not overloaded"; a **startupProbe** covers slow initialization.

**Follow-ups**
- What happens if readiness fails on *all* pods at once? *(The Service has zero endpoints: a total outage, although nothing restarts.)*
- What liveness timeouts would you use for a CPU-heavy inference service, and why generous ones?

### Q7.3 ★★ — CPU and memory requests vs limits: what happens when a container exceeds each?
**Key points**
- **Requests** are used for scheduling and guarantees, and they are the **baseline for HPA utilization percentages**.
- Exceed the **CPU limit** and you get **throttling** (CFS quota): latency spikes, no kill.
- Exceed the **memory limit** and you get **OOMKilled** (exit code **137**) and a restart.
- QoS classes: Guaranteed (requests = limits), Burstable, BestEffort. These affect eviction order under node pressure.

**Follow-ups**
- Why do some teams set CPU requests but *no* CPU limits? *(To avoid throttling-induced tail latency while keeping scheduling fairness.)*
- Why should memory requests usually equal memory limits for inference pods?

### Q7.4 ★★ — HPA targets 70% CPU, yet p95 latency spikes during every traffic burst. Why doesn't autoscaling save us?
**Key points**
- **Reaction lag** adds up: metrics-server scrape (about 15 s), the HPA sync period (15 s), the scale decision, pod scheduling, **image pull**, container start, **model load**, and readiness. That is often **30–120 s**.
- The burst arrives before the new capacity does.
- Utilization is measured against *requests*. With a wrong request value, the percentages mislead.
- Fixes: higher `minReplicas` (headroom), a lower target (50%), scaling on RPS, queue length or latency (custom metrics or KEDA), smaller images, pre-pulled images, faster model loading, predictive or scheduled scaling.

**Follow-ups**
- Compute: 3 replicas at 140% of request CPU, target 70%. What does the HPA want? *(ceil(3 × 140/70) = 6.)*
- Why does the HPA scale *down* slowly by default? *(The 5-minute stabilization window avoids flapping.)*

### Q7.5 ★★ — What belongs in a ConfigMap, what in a Secret, and what in the image?
**Key points**
- **Image**: code, dependencies, and (for small models) weights. It is immutable.
- **ConfigMap**: non-sensitive config such as `MODEL_VARIANT=int8`, thread counts, log level, `API_URL`.
- **Secret**: API keys, DB passwords, registry credentials.
- Secrets are only **base64-encoded** by default, not encrypted. Enable encryption at rest, use RBAC, and consider external secret managers.

**Follow-ups**
- You edit a ConfigMap. Do running pods see the change? *(Environment variables: no, until restart. Mounted files: eventually, but the app must re-read them.)*
- How would you trigger a rollout on config change? *(A checksum annotation in the pod template.)*

### Q7.6 ★ — Why use a Deployment instead of creating Pods directly?
**Key points**
- Bare pods are not rescheduled when they die or a node fails.
- A Deployment manages ReplicaSets and gives desired replica count, self-healing, rolling updates, rollback history and scaling.

**Follow-ups**
- What does a ReplicaSet do that a Deployment adds to? *(An RS keeps N pods running. A Deployment manages RS versions for rollouts.)*
- When would you use a StatefulSet or a Job instead? *(Stable identity or storage; batch inference or training.)*

### Q7.7 ★★ — ClusterIP vs NodePort vs LoadBalancer vs Ingress: which for the API and which for Streamlit?
**Key points**
- The **API** may only need **ClusterIP** if only Streamlit calls it (internal, not exposed = smaller attack surface).
- **Streamlit** faces users, so use a LoadBalancer or an Ingress (one entry point, TLS, path routing). NodePort is fine for minikube or kind demos.
- On minikube, `LoadBalancer` stays `<pending>` without `minikube tunnel`, because there is no cloud load balancer controller.

**Follow-ups**
- If the Streamlit browser code (not the server) called the API, would ClusterIP still work? *(No. Browsers are outside the cluster. Streamlit calls from its server side, so it works.)*
- How does k3s expose services on an Oracle VM? *(It ships Traefik ingress and ServiceLB. Open the VM's security list and firewall too.)*

### Q7.8 ★★ — Traffic is low, so you run 1 replica. Why is that risky even with HPA configured?
**Key points**
- It is a single point of failure: node drains, rollouts (with `maxUnavailable` rounding), OOM restarts and liveness restarts all mean downtime.
- There is no warm capacity when a burst arrives (see Q7.4).
- Use at least 2 replicas across nodes (anti-affinity or topology spread) and a **PodDisruptionBudget**.

**Follow-ups**
- What does a PDB protect against, and what does it *not*? *(Voluntary disruptions such as drains. Not node crashes.)*
- On a single-node k3s VM, does replicas: 2 add availability? *(Only against pod-level failure, not node failure.)*

### Q7.9 ★★★ — The model is 800 MB. Compare baking it into the image, an init container download, and a shared PersistentVolume, in terms of scale-up speed.
**Key points**
- **Baked in**: every new node pulls the big image (slow the first time, then cached). Simple and immutable.
- **Init container from object storage**: a small image, but every pod downloads 800 MB (network cost, slower per pod, dependency on the bucket).
- **Shared PV (ReadOnlyMany)** or node-local cache: fast after the first load, but more infrastructure and access-mode limits (many block volumes are ReadWriteOnce).
- Also consider image pre-pulling or DaemonSets, lazy-loading snapshotters, and smaller models (quantize to shrink 4×).

**Follow-ups**
- Which option gives the best *reproducibility*?
- How would quantizing the model change this whole decision? *(800 MB → 200 MB makes baking it in easy.)*

### Q7.10 ★★ — Why is `image: myapi:latest` a bad idea in Kubernetes manifests?
**Key points**
- It is not reproducible: which build is "latest"? Rollback is ambiguous, and different nodes may cache different "latest" images.
- `kubectl apply` with an unchanged manifest (still `:latest`) **doesn't trigger a rollout**, because the pod spec didn't change.
- `:latest` implies `imagePullPolicy: Always`. Use immutable tags (git SHA, semver) or digests.

**Follow-ups**
- How does using the git SHA as the tag help incident response?
- What does `imagePullPolicy: IfNotPresent` plus a reused tag do to your update?

### Q7.11 ★★★ — Why can't two pods share one GPU by default in Kubernetes, and what does that mean for cost?
**Key points**
- The NVIDIA device plugin exposes `nvidia.com/gpu` as an **integer, whole-device** resource, so each pod gets whole GPUs.
- A small model using 10% of a T4 still pays for 100%.
- Options: time-slicing (shared, no isolation), MIG (hardware partitions on A100/H100-class GPUs), MPS, or batching many models or requests into one server (Triton).

**Follow-ups**
- For MobileNetV2 at our traffic levels, is a GPU node justified at all? *(Almost never. See Fermi F4.)*
- When does GPU serving become cheaper per prediction than CPU? *(High sustained throughput, big models, batching.)*

---

## 8. Cloud & Cost

### Q8.1 ★★ — Serving 1M image classifications per day: CPU or GPU?
**Key points**
- 1M/day is about **11.6 req/s average**, maybe 35–60 req/s at peak.
- MobileNetV2 (int8 ONNX, 224 px) takes roughly 5–15 ms per image per core on modern x86, so a **few vCPUs** handle it.
- A GPU instance (a T4 VM is roughly $0.35–0.55/hr for the GPU alone, varying by provider and region) would sit mostly **idle**, because batch-1, small-model inference is dominated by overheads.
- A GPU wins with large models, high sustained throughput, or when batching is possible. See Fermi F4 for numbers.

**Follow-ups**
- At what traffic level does the GPU break even? What assumptions drive that number?
- What changes if the model were ViT-L or a diffusion model?

### Q8.2 ★ — Why does the free Render service take 30–60 s to answer the first request after a quiet period?
**Key points**
- Free web services **spin down after about 15 minutes idle**. The next request triggers a **cold start**: schedule the container, pull or unpack the image, start Python, **load the model**.
- Mitigations: smaller image (ORT instead of torch), fast model load, a UX message ("waking up…"), a paid always-on tier for real users.
- Keep-alive pings may violate the provider's terms and defeat the purpose of a free tier. Discuss this honestly.

**Follow-ups**
- Which parts of cold start do *you* control? *(Image size, import time, model load time, lazy imports.)*
- Hugging Face Spaces (CPU Basic) also sleeps after inactivity, and Docker Spaces now need a paid PRO plan. How would you communicate these limits in a demo?

### Q8.3 ★★ — Oracle Cloud's Always Free tier gives an Ampere ARM VM (2 OCPUs / 12 GB total since June 2026; many blogs still say 4/24). What changes for your deployment?
**Key points**
- **arm64 architecture**: you need arm64 or multi-arch Docker images, built with buildx or natively on the VM.
- Wheel availability: onnxruntime, torch, numpy and pillow publish aarch64 wheels, but check niche packages.
- Performance characteristics differ: int8 paths (ARM dot-product instructions), PyTorch quantization engine `qnnpack`, different thread scaling.
- Generous RAM makes k3s plus several services feasible for free.

**Follow-ups**
- You built `linux/amd64` on your laptop and ran it on the Ampere VM. What error do you get? *(`exec format error`.)*
- What is a multi-arch manifest?

### Q8.4 ★★ — Where do surprise cloud bills come from on ML projects?
**Key points**
- **Idle GPUs** (a notebook left running over the weekend), forgotten clusters and node pools, load balancers (billed per hour), **egress** (data leaving the cloud), NAT gateways, persistent disks and snapshots left after deleting VMs, public IPs, log ingestion and retention, container registry storage, cross-region traffic.
- Guardrails: budgets and alerts, auto-shutdown schedules, quotas, labels and tags for cost attribution, deleting resources via IaC (`terraform destroy`).

**Follow-ups**
- Why is egress usually priced higher than ingress?
- A student gets a $300 credit. What is the fastest way to burn it by accident? *(A GPU VM or a managed K8s cluster left running.)*

### Q8.5 ★★ — Managed PaaS (HF Spaces, Render, Cloud Run) vs VM + k3s vs managed Kubernetes (GKE, AKS): how do you choose?
**Key points**

| Option | Best when | Pay with |
|---|---|---|
| HF Spaces / Render | Demo, portfolio, low traffic, tiny team | Cold starts, limited control and resources |
| Cloud Run / serverless containers | Spiky traffic, scale-to-zero, pay per request | Cold starts, per-request limits |
| VM + Docker/k3s | Full control, fixed small budget, learning | You operate everything (patching, TLS, backups) |
| Managed K8s | Many services, team, SLOs, autoscaling | Cost (control plane and nodes), complexity |

- The decisive factors are traffic shape, team size and ops skill, budget, compliance, and latency needs.

**Follow-ups**
- What would you pick for a university demo used 2 hours per week? For a startup with 3 engineers and 50k users?
- What is the hidden cost of "free" self-hosting? *(Engineer time and on-call.)*

### Q8.6 ★★ — Serverless (Cloud Run, Lambda) for ML inference: pros and cons?
**Key points**
- Pros: scale to zero, per-request billing, no servers to manage, automatic scaling.
- Cons: cold starts (image pull plus model load), memory and CPU caps per instance, request timeouts, limited or no GPU on some platforms, and a per-request price that exceeds a VM at high steady load.
- Works well for small models (MobileNet ONNX), bursty low-average traffic, and async batch jobs.

**Follow-ups**
- What model size makes serverless painful? *(Hundreds of MB to GBs, since load time dominates.)*
- How does "min instances = 1" change the cost and cold-start picture?

### Q8.7 ★★ — A user in India calls your API hosted in us-east. Model inference takes 20 ms. What latency does the user see?
**Key points**
- Great-circle distance is about 13,000 km. Light in fibre travels at about 200,000 km/s, so the **minimum RTT is about 130 ms**. Real routes give roughly **200–300 ms RTT**.
- TCP plus TLS handshakes cost 1–3 more RTTs on a new connection, and uploading a large image adds transfer time.
- The 20 ms model is a **small fraction**. Optimizing it from 20 to 10 ms is invisible to this user. Moving to an Asian region is not.

**Follow-ups**
- List three things that help more than a faster model here. *(Nearby region or edge, keep-alive and HTTP/2, client-side resize to shrink the upload.)*
- When *does* model latency dominate? *(Big models, or co-located clients.)*

### Q8.8 ★★ — Spot or preemptible instances: great for training, risky for inference. Why?
**Key points**
- Spot instances are often 60–90% cheaper but **can be reclaimed at short notice** (about 30 s to 2 min).
- **Training** can checkpoint and resume, so interruptions only cost a little progress.
- **Inference** needs availability. Use a mix: an on-demand baseline plus spot for extra capacity, spread across instance types and zones, with graceful draining.

**Follow-ups**
- What must you checkpoint to resume training exactly? *(Model, optimizer, LR scheduler, epoch or step, RNG states, data-loader position.)*
- How often should you checkpoint? (A trade-off between I/O cost and lost work.)

### Q8.9 ★★★ — How do you compute cost per 1,000 predictions, and which levers reduce it?
**Key points**
- **Cost per 1k = (instance $/hr) ÷ (sustained predictions/hr at target utilization) × 1000.**
- Levers: faster model (quantization, smaller resolution, distillation), more throughput per instance (thread tuning, batching), cheaper hardware (ARM, spot for batch work), right-sizing, autoscaling to follow demand, caching repeated inputs, and avoiding over-provisioned idle time.
- Target about **60–70% utilization**, not 100%: queueing theory says latency grows sharply as utilization approaches 1.

**Follow-ups**
- Why does latency blow up near 100% utilization? *(In an M/M/1 queue, the wait is proportional to ρ/(1−ρ).)*
- Cost per prediction vs cost per *correct* prediction: when does a more expensive but more accurate model win?

### Q8.10 ★★ — Your free HF Space demo works fine, but crashes or freezes when 30 students open it at once. Why, and what can you do on a free tier?
**Key points**
- One small container with no autoscaling. Inference runs serially, requests queue, timeouts follow. Memory spikes from concurrent image decoding cause OOM.
- Streamlit reruns the script per interaction. If the model is not cached with `st.cache_resource`, each session reloads it.
- Free-tier mitigations: cache the model, downscale inputs early, use a lighter model (int8 ONNX), a queue with a concurrency limit (Gradio's queue), rate limiting, a clear "busy" message.

**Follow-ups**
- Estimate: 30 users × 1 request per 10 s at 100 ms per inference on 2 vCPUs. Is it overloaded? *(3 req/s × 0.1 s = 0.3 core-seconds per second, so no. The problem is usually reloading or memory, not compute.)*
- What does this teach about load testing before a demo? *(Use locust or k6.)*

---

## 9. MLOps, Monitoring & Data Drift

### Q9.1 ★★★ — "The model is 95% accurate in the notebook, but users complain it's wrong." List 10 possible reasons.
**Key points** (a strong answer spans data, model, serving and people)
1. **Preprocessing mismatch** between training and serving (RGB/BGR, mean/std, resize, `/255`).
2. **Data drift**: users' photos (lighting, camera, background, resolution) differ from training data.
3. **Open-set inputs**: users submit things outside the trained classes, and the model still picks one confidently.
4. **Inflated test accuracy**: leakage or near-duplicates between train and test, or tuning on the test set.
5. **Class imbalance**: 95% overall hides poor recall on the classes users care about.
6. **Subgroup failure**: good on average, bad for some users, devices or regions.
7. **Wrong artefact deployed**: old version, the int8 variant degraded, a broken export, `eval()` forgotten before export.
8. **Latency or timeouts** that users perceive as "wrong" or broken.
9. **Overconfident wrong answers** (poor calibration), which feel worse than uncertain ones, or the threshold is set badly.
10. **EXIF rotation, compression, screenshots** instead of photos.
11. **Mismatch with expectations**: users want top-3, or a class the product promised but training didn't cover.
12. **Selection bias in complaints**: only failures get reported, so 5% of 100k requests is 5,000 unhappy people.

**Follow-ups**
- Which three would you check *first*, and what is the cheapest test for each? *(A golden image through the API; log and compare input statistics; confirm the deployed version hash.)*
- How do you get ground-truth labels in production? *(User feedback buttons, sampling for human labelling, delayed labels.)*

### Q9.2 ★★ — Data drift vs concept drift vs label shift: define each with an example from our projects.
**Key points**
- **Data (covariate) drift**: P(x) changes, P(y|x) the same. The pedestrian camera moves from daytime campus to night-time streets.
- **Concept drift**: P(y|x) changes. What counts as a "defect" gets redefined by quality control, or fashion styles change what a "shirt" looks like.
- **Label (prior) shift**: P(y) changes. Winter means more coats and fewer sandals.

**Follow-ups**
- Which of these can you detect **without labels**? *(Data drift fully; label shift partly, from the prediction distribution; concept drift generally needs labels.)*
- Which one does retraining on fresh data fix, and which needs relabelling?

### Q9.3 ★★ — How would you detect drift in the Day 3 image API without any labels?
**Key points**
- Monitor **input statistics**: image size, aspect ratio, brightness, contrast, blur, colour histograms.
- Monitor **embeddings**: backbone features compared with a training reference, using MMD, a domain classifier, or the distance to the nearest training cluster.
- Monitor **outputs**: the predicted-class distribution (PSI, chi-square) and the confidence distribution. A drop in average confidence is a useful early warning.
- Use windowed comparisons against a reference window and alert thresholds tuned to avoid alert fatigue.

**Follow-ups**
- Is a drift alert a reason to retrain automatically? *(No. Investigate first. Drift doesn't always hurt accuracy.)*
- How would you pick the threshold to avoid false alarms? *(Backtest on historical windows, and use rules like "persist for N windows".)*

### Q9.4 ★★ — Why should softmax scores not be treated as calibrated probabilities?
**Key points**
- Softmax outputs sum to 1 and *look* like probabilities, but nothing forces "0.9 confidence" to mean "right 90% of the time".
- Modern deep networks trained with cross-entropy tend to be **overconfident** (Guo et al., 2017). Longer training, large capacity and little regularization make it worse.
- **Out-of-distribution** inputs (a photo of a dog sent to the FashionMNIST model) still get high softmax scores. Softmax only compares classes against each other; it can't say "none of the above".
- Measure with **reliability diagrams** and **ECE** (Expected Calibration Error).
- Fix with **temperature scaling** (fit one scalar T on validation data, logits/T), label smoothing, ensembles, or Platt or isotonic regression.

**Follow-ups**
- Why doesn't temperature scaling change accuracy? *(Dividing by T > 0 keeps the order of the logits, so the argmax is unchanged.)*
- Where does calibration matter operationally? *(Confidence thresholds for automation vs human review, triage (Case 3), cost-sensitive decisions, combining models.)*

### Q9.5 ★★ — Design the monitoring dashboard for the Day 3/4 API. What panels, and what alerts?
**Key points**
- **Service health (RED)**: Rate (req/s), Errors (5xx and 4xx rate), Duration (p50/p95/p99 latency).
- **Saturation**: CPU (including throttling), memory versus limit, pod restarts, replica count (HPA).
- **Model health**: predicted-class distribution, mean confidence, share of low-confidence predictions, input statistics and drift scores, variant and version distribution.
- **Business or quality**: user feedback (thumbs down rate), labelled-sample accuracy when labels arrive.
- Alert on symptoms (p95 above the SLO, 5xx above 1%, restarts) rather than every metric.

**Follow-ups**
- Which of these can Prometheus scrape from a `/metrics` endpoint, and which need a batch job?
- What is an SLO and an error budget? *(For example 99.5% of requests under 300 ms over 30 days.)*

### Q9.6 ★★ — What exactly must you version to reproduce a model six months later?
**Key points**
- **Code** (git SHA), **data** (dataset snapshot or hash, DVC, split definitions), **config and hyperparameters**, **environment** (Docker image or lockfile, CUDA and driver versions), **random seeds**, **pretrained weights version** (the torchvision weights enum), **preprocessing** parameters, and the evaluation script.
- The artefact goes into a **model registry** (MLflow or similar) with metrics and lineage.

**Follow-ups**
- A torchvision upgrade changed the default pretrained weights. How would you have noticed? *(Pin the weights enum, such as `MobileNet_V2_Weights.IMAGENET1K_V1`, and record it.)*
- Is bit-exact reproducibility necessary, or is "within noise" enough?

### Q9.7 ★★ — Shadow deployment vs canary vs A/B test: when would you use each?
**Key points**
- **Shadow**: the new model gets a copy of live traffic and its outputs are *not* shown to users. Zero user risk. Compare agreement, latency and errors.
- **Canary**: a small share of real traffic (1–10%) goes to the new version, with automatic rollback on bad metrics. Its purpose is safety.
- **A/B test**: a randomized split to measure *business impact* with statistical rigour. Its purpose is decision-making.

**Follow-ups**
- What does shadowing cost? *(About 2× compute for the mirrored share.)*
- How long should an A/B test run? What is "peeking" and why is it a problem?

### Q9.8 ★★ — Should you retrain on a schedule or when a trigger fires? What can go wrong with automatic retraining?
**Key points**
- Schedule (weekly) is simple and predictable. A trigger (drift or performance drop) is efficient but needs reliable monitoring. Many teams combine them.
- Risks: training on **unlabelled or badly labelled** fresh data, **feedback loops** (Q9.9), **data poisoning** by malicious users, silent regressions if there is no evaluation gate.
- Always have an **evaluation gate**: the new model must beat the current one on a fixed benchmark *and* on recent data, with a per-class check, before promotion.

**Follow-ups**
- What is in the "fixed benchmark", and why must it never be trained on?
- Who approves promotion: a human or the pipeline?

### Q9.9 ★★★ — Explain a feedback loop where a model's own predictions corrupt its future training data. How do you break it?
**Key points**
- Example: a recommender only shows items it predicts are good, so it only ever gets feedback on those items and "confirms" itself.
- Example: a defect detector auto-accepts parts under its threshold, and only rejected parts get human-inspected and labelled. The training data then under-samples missed defects.
- Example: predictive policing sends patrols where the model predicts crime, more crime gets recorded there, and the model is "confirmed".
- Break it: **exploration** (randomly route a small share to human review regardless of the prediction), log the model's decision alongside the label, counterfactual evaluation, audit sampling.

**Follow-ups**
- What share of traffic would you randomly send to human review, and how do you justify the cost?
- Which of our five system-design cases is most exposed to feedback loops?

### Q9.10 ★★ — Offline accuracy improved by 2%, but the online business metric didn't move. Why?
**Key points**
- The offline test set doesn't represent production (drift, different class mix).
- The improved classes aren't the ones that matter to users or the business.
- The metric mismatch: accuracy vs what users experience (latency, top-3, confidence UX).
- Measurement noise: the online test is underpowered.
- The model isn't the bottleneck: UX, latency or the upstream data dominate.

**Follow-ups**
- How would you design an offline metric that better predicts the online one?
- What if the new model is 2% better but 40% slower. Ship it?

### Q9.11 ★★ — What automated tests belong in CI/CD for an ML inference service?
**Key points**
- **Unit tests**: preprocessing (shapes, dtype, value ranges, RGB order), input validation, error codes.
- **Golden tests**: fixed images give expected top-1 and logits within tolerance.
- **Export parity**: ONNX vs PyTorch outputs within about 1e-4 relative, top-1 agreement on a sample set; int8 agreement above a threshold.
- **Model quality gate**: accuracy (and per-class accuracy) on a held-out set is at least a threshold.
- **Performance regression**: latency p95 on reference hardware, image size budget.
- **Container smoke test**: build the image, start it, call `/health` and `/predict`.
- **Manifest checks**: `kubectl --dry-run`, lint, policy (no `:latest`, resources set).

**Follow-ups**
- Which of these would have caught the "forgot `model.eval()` before export" bug? *(Export parity and the golden test.)*
- Which tests are flaky by nature, and how do you handle them? *(Latency tests: use generous thresholds and dedicated runners.)*

---

## 10. Ethics & Responsible AI

### Q10.1 ★★ — Penn-Fudan: 170 images from campus and urban streets around two US universities. If this detector were used in a real vehicle or camera system, who or what is underrepresented?
**Key points**
- Conditions: night, rain, fog, snow, glare, motion blur, unusual angles.
- People: children, wheelchair users, people carrying large objects, cyclists, crowds, heavy occlusion, varied clothing (cultural dress, umbrellas), and a range of skin tones under different lighting.
- 170 images cannot cover this. Failures are safety-critical and **unequally distributed**.
- Needed: disaggregated evaluation (recall per condition and subgroup), targeted data collection, documented limitations (model cards, datasheets).

**Follow-ups**
- How would you measure recall for "people in wheelchairs" if you only have 3 examples? *(You can't reliably. That in itself is a finding: collect more data.)*
- Who is responsible when the detector misses someone: the data team, the model team, or the product owner?

### Q10.2 ★★ — Overall accuracy is 95%, but it's 72% for one subgroup. How would you find this, and what are your options?
**Key points**
- Find it: **slice-based evaluation** on metadata (device, region, lighting, demographic attributes where lawful and consented), error analysis clustering, feedback by segment.
- Options: collect more data for that group, reweight or resample, group-aware thresholds (legally and ethically sensitive), a different model, human review for that slice, or **don't deploy** for that use case. Document it in a model card.
- Fairness metrics can conflict (equal accuracy vs equal false-positive rates vs calibration within groups). You often **cannot satisfy all of them at once**.

**Follow-ups**
- If you don't collect demographic attributes (for privacy), how can you check fairness?
- Is it acceptable to ship with the gap disclosed? Who decides?

### Q10.3 ★★ — You're building a GAN-based avatar generator app. What are the risks, and what mitigations would you build in?
**Key points**
- Risks: **deepfakes and impersonation** (if the user uploads someone else's photo), **consent**, harassment, training-data **copyright and privacy** (were the faces scraped?), biased outputs (poor quality for some groups, stereotyped styles), minors, storage of biometric data.
- Mitigations: consent and terms, face-match or liveness check that the uploader is the subject, **watermarking and provenance** (C2PA content credentials), abuse reporting, not keeping uploaded photos (delete after generation), data minimization, age gating, bias evaluation across skin tones and genders.

**Follow-ups**
- Can watermarks be removed? Does that make them useless? *(Robustness is limited, but they still raise the cost of misuse and help honest platforms.)*
- Which laws matter? *(GDPR biometric data rules, India's DPDP Act 2023, the EU AI Act transparency obligations for deepfakes.)*

### Q10.4 ★★ — A chest X-ray triage model: who chooses the decision threshold, and how should the trade-off between false negatives and false positives be made?
**Key points**
- A false negative (missed disease, delayed care) usually costs far more than a false positive (extra radiologist review). The threshold should favour **high sensitivity**, constrained by radiologist capacity.
- It is a **clinical and organisational decision**, made with clinicians, using cost analysis. Engineers provide the ROC/PR curves and calibrated probabilities.
- The model **prioritizes**; it does not diagnose. Regulatory status (a medical device) matters.

**Follow-ups**
- Why does calibration (Q9.4) matter here more than raw accuracy?
- What happens to the threshold if prevalence differs between the training hospital and the deployment hospital? *(PPV changes. Recalibrate locally.)*

### Q10.5 ★★ — You want to log user-uploaded images to improve the model. What must be in place first?
**Key points**
- **Legal basis and informed consent** (clear opt-in), a **purpose limitation**, a **retention policy** (delete after N days), **access control** and encryption, anonymization (faces or text in images), the right to deletion, and a data processing agreement if you use a cloud provider.
- Relevant laws: GDPR and **India's Digital Personal Data Protection Act 2023**, and sector rules (health data).
- Engineering: separate storage, audit logs, no raw images in general application logs.

**Follow-ups**
- Can you improve the model *without* storing raw images? *(Store features or embeddings — though these can leak too — or federated learning, or synthetic data.)*
- What is the privacy risk of a model memorizing training images, especially for GANs?

### Q10.6 ★★ — Can quantization or pruning be a fairness issue even when top-line accuracy barely changes?
**Key points**
- Yes. Research ("What Do Compressed Deep Neural Networks Forget?", Hooker et al., 2019) shows compression **disproportionately hurts rare or long-tail classes and underrepresented examples**, while overall accuracy looks unchanged.
- The model "forgets" what it saw least. That often matches minority groups or rare-but-important cases.
- So: compare **per-class and per-slice** metrics before and after compression, not just overall accuracy.

**Follow-ups**
- Which CIFAR-10 classes would you expect to degrade most after int8, and how would you check?
- Add this check to the Q9.11 CI pipeline. What would the assertion look like?

### Q10.7 ★★ — Energy and carbon: for a widely deployed model, does training or inference dominate lifetime energy use? How does Day 2 optimization help?
**Key points**
- Training is a large one-time cost. **Inference** runs millions or billions of times, so for popular models the lifetime inference energy often **exceeds** training energy.
- Optimization (int8, pruning, distillation, lower resolution) cuts energy per prediction, sometimes by 2–4×. Right-sizing hardware and autoscaling cut idle waste.
- Region choice matters (grid carbon intensity).

**Follow-ups**
- How would you estimate energy per prediction? *(Device power × latency. See Fermi F9.)*
- Is edge inference greener than the cloud? It depends on what?

### Q10.8 ★★ — A user asks, "Why did your model reject my image?" What can you honestly tell them?
**Key points**
- Post-hoc explanation tools (**Grad-CAM**, SHAP, integrated gradients, occlusion) show *where* the model looked. They are approximate, can be unstable, and can **mislead**, suggesting reasoning that isn't there.
- Honest answers combine the prediction, a calibrated confidence, known limitations, and a path to appeal or human review.
- In high-stakes domains, a right to explanation or contestation may be legally required.

**Follow-ups**
- How could you check whether a Grad-CAM map is faithful? *(Sanity checks: randomize the weights and see whether the map changes. Deletion or insertion metrics.)*
- Can explanations reveal a spurious correlation (for example, the model uses hospital-specific text markers on X-rays)?

### Q10.9 ★★★ — "A human always reviews the model's output, so it's safe." Why might that be false?
**Key points**
- **Automation bias**: people over-trust and rubber-stamp confident machine suggestions, especially under time pressure or high volume.
- **Deskilling** over time. Reviewers see only what the model surfaces, so they cannot catch false negatives the model filtered out.
- Mitigations: show calibrated uncertainty, sometimes hide the model's suggestion (blind review), audit the reviewers, measure human+AI performance together, set workload limits.

**Follow-ups**
- In the X-ray triage case (Case 3), how would you design the UI to reduce automation bias?
- How would you measure whether the human is adding value? *(Override rate, and accuracy of overrides.)*

### Q10.10 ★★ — What security threats face a public model API beyond ordinary web security?
**Key points**
- **Model extraction or stealing** through many queries (returning full probability vectors makes it easier).
- **Adversarial examples**: tiny perturbations that cause wrong predictions.
- **Membership inference**: was this person's image in the training set?
- **Data poisoning** via feedback or retraining loops.
- Resource abuse: huge uploads, decompression bombs, request floods, which cost money on autoscaling infrastructure.
- Mitigations: authentication, rate limiting, return top-k labels instead of full logits, input validation, anomaly detection on query patterns, cost caps.

**Follow-ups**
- Why can autoscaling turn a DoS attack into a *billing* attack? How do you cap it? *(`maxReplicas`, budgets.)*
- Is returning confidence scores a privacy or security trade-off?

### Q10.11 ★★★ — Your manager says: "Ship it. We'll fix the bias issues in v2." How do you respond?
**Key points**
- Quantify the harm: who is affected, how badly, how reversible. Is this a demo, or safety- or rights-relevant?
- Options besides "ship" or "don't ship": limited rollout, human review for affected slices, restricting the use case, clear disclosure of limitations, monitoring with a rollback trigger, a committed timeline for v2.
- Professional responsibility: document the decision and who made it (model card). Know the regulatory context (EU AI Act high-risk categories).

**Follow-ups**
- What evidence would change your manager's mind fastest? *(Disaggregated metrics, incident scenarios, legal exposure.)*
- When is it legitimate to ship with known limitations?
