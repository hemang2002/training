# Day 2 — Transfer Learning & Model Optimization

**Course:** Train → Optimize → Serve → Containerize → Deploy ML models with PyTorch
**Audience:** MSc students · **Duration:** 5 h (including two breaks and ~2 h of hands-on lab)
**Hardware assumption:** laptop CPU, no GPU needed.

> This file is the **trainer's lecture guide**. The sections under "Concepts" are written so you can
> speak them almost verbatim; the ⏱ markers tell you where you are in the timetable.

---

## 1. Learning objectives

By the end of the day students can …

1. explain **why transfer learning works** and choose between *feature extraction* and *fine-tuning*;
2. fine-tune an ImageNet-pretrained **MobileNetV2** on CIFAR-10 on a CPU and save it correctly (`state_dict`);
3. explain number formats (FP32, FP16, BF16, INT8) and the **affine quantization** mapping (scale, zero-point);
4. apply and compare **dynamic quantization, static post-training quantization (PTQ), QAT, FP16 and pruning**,
   and argue which one fits which model and hardware;
5. explain why **unstructured pruning does not speed up dense kernels**;
6. **export to ONNX**, validate it, run it with **ONNX Runtime**, quantize it to INT8 and benchmark it;
7. produce the **model artefacts + model card** that Day 3's Flask API will serve.

## 2. Folder contents

| File | Purpose | Runtime (CPU) |
|---|---|---|
| `01_transfer_learning_mobilenetv2.ipynb` | Feature extraction vs fine-tuning, final model → `models/` | ~7–10 min (`QUICK_RUN`: ~1 min) |
| `02_quantization_and_pruning.ipynb` | Baseline, dynamic / static PTQ / FP16 / BF16 / pruning / QAT, comparison charts | ~4 min |
| `03_onnx_export_and_benchmark.ipynb` | ONNX export + validation, ORT graph optimisations, static INT8 ONNX, `model_card.json` | ~2 min |
| `04_student_lab.ipynb` | 6 exercises with TODOs + hidden solutions | ~5 min for all solutions |
| `scripts/train_and_export.py` | **One-command fallback**: the whole Day-2 pipeline → `models/` | ~6 min |
| `outputs/` | created by the notebooks (CSV results, temporary ONNX files) | – |

All notebooks have a `QUICK_RUN` flag in the first code cell (tiny subsets, for a smoke test). Paths are
relative to this folder: data in `../data`, models in `../models`.

**Before the day (trainer checklist)**

```bash
# from the course root, with the course virtual environment active
python Day2_Transfer_Learning_and_Optimization/scripts/train_and_export.py      # ~6 min, downloads CIFAR-10 once
```

This guarantees that `models/` contains a complete, consistent set of Day-3 artefacts even if a student's
training run fails. In Jupyter select the **course venv kernel** (the one with `torch 2.9.x+cpu`) — the first
cell of every notebook prints the torch version; check it.

**Artefacts produced in `../models/` (contract with Day 3 — do not rename):**

| File | Content |
|---|---|
| `mobilenetv2_cifar10_fp32.pth` | `state_dict` of torchvision `mobilenet_v2` with `classifier[1] = Linear(1280, 10)` |
| `model_fp32.onnx` | FP32 ONNX, opset 17, input `input` `[batch,3,96,96]`, output `logits` `[batch,10]` (raw logits) |
| `model_int8.onnx` | static INT8 ONNX (QDQ, per-channel, calibrated) — same interface |
| `labels.json` | the 10 CIFAR-10 class names in index order |
| `model_card.json` | `img_size`, `mean`, `std`, `labels`, tensor names, per-variant size/accuracy/latency, creation date |

Inference preprocessing (identical in training evaluation and in the API):
`PIL RGB → Resize((96,96)) → ToTensor → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) → NCHW float32`.

---

## 3. Timetable (5 h = 300 min)

Relative times (T+0 = start of the session). Lab blocks = **115 min (38 %)**.

| Time | Min | Type | Content | Material |
|---|---|---|---|---|
| T+0:00 | 10 | Lecture | Welcome, recap Day 1 (training loop, CIFAR-10), today's goal: *a model small and fast enough to serve tomorrow*. **Start `train_and_export.py` or NB01 in the background now.** | this README §1 |
| T+0:10 | 25 | Lecture | Transfer learning: why it works, feature extraction vs fine-tuning, freezing, ImageNet normalisation, augmentation | §4.1 |
| T+0:35 | 20 | Live-code | Walk through Notebook 01 (the run started at T+0:00 now has results) | NB01 |
| T+0:55 | 40 | **STUDENT LAB A** | Students run NB01 with `QUICK_RUN=True` (then full, in the background), then **Lab Ex. 1 (backbones, 20 min)** and **Ex. 2 (freezing, 15 min)** | NB01, NB04 |
| T+1:35 | 10 | Break | | |
| T+1:45 | 35 | Lecture | Why optimise; number formats; quantization theory (affine mapping, per-tensor/per-channel); dynamic vs static PTQ vs QAT; pruning; distillation | §4.2 |
| T+2:20 | 20 | Live-code | Notebook 02: manual scale/zero-point demo, static PTQ step by step, pruning, summary chart | NB02 |
| T+2:40 | 35 | **STUDENT LAB B** | Students run NB02 and fill in the results table (§5); **Lab Ex. 3 (calibration data, 15 min)** and **Ex. 4 (structured pruning, 15 min)** | NB02, NB04 |
| T+3:15 | 10 | Break | | |
| T+3:25 | 15 | Lecture | TorchScript vs ONNX, ONNX Runtime & execution providers, graph optimisations / operator fusion, why we serve ORT tomorrow | §4.3 |
| T+3:40 | 15 | Live-code | Notebook 03: export, validate, fusion levels, INT8 ONNX, model card, NumPy-only inference | NB03 |
| T+3:55 | 40 | **STUDENT LAB C** | Students run NB03; **Lab Ex. 5 (threads, 10 min)** and **Ex. 6 (batch size, 10 min)**; complete the results table; buffer for unfinished exercises | NB03, NB04 |
| T+4:35 | 25 | Discussion | Results table comparison across students' laptops, discussion questions (§6), pitfalls (§7), recap & homework (§8), preview Day 3 | §5–§8 |

Tips: if training is slow on student laptops, let them use the trainer's `models/` (copied from the script run)
and run NB01 in `QUICK_RUN` mode only — Notebooks 02–04 only need the files in `models/`.

---

## 4. Concepts — speakable explanations

### 4.1 Transfer learning (T+0:10, 25 min)

**Why transfer learning?**
"Training a CNN from scratch on ImageNet takes 1.28 million labelled images and days of GPU time. Most of what
the network learns is not specific to ImageNet's 1000 classes: the first layers learn edges and colour blobs, the
middle layers textures and patterns, the later layers object parts. These features are useful for almost any
natural-image task. Transfer learning means: **reuse those learned features and only teach the network the new
task**. You need far less data, far less compute, and you usually get better accuracy than training from
scratch on a small dataset — today we reach ~90 % on CIFAR-10 with 10 000 images and 6 CPU minutes."

**Anatomy.** "A classification CNN is a *backbone* (feature extractor) plus a *head* (classifier). In
MobileNetV2 the backbone is `model.features` and the head is `model.classifier = Dropout → Linear(1280, 1000)`.
We replace only the last layer by `Linear(1280, 10)`. That new layer is random; everything else is pretrained."

**Feature extraction vs fine-tuning.**

| | Feature extraction | Fine-tuning |
|---|---|---|
| What is trained | only the new head | head + (part of) the backbone |
| Trainable params (MobileNetV2) | 12 810 | 2.2 M |
| Compute per step | low (no backbone gradients) | higher |
| Data needed | very little (hundreds) | more (thousands+) |
| Accuracy on a shifted domain | limited | best |
| Learning rate | higher (e.g. 3e-3) | lower (e.g. 1e-3, or smaller for backbone) |

"Feature extraction treats the backbone as a fixed function. Fine-tuning lets the features adapt — CIFAR images
are tiny and blurry compared to ImageNet photos, so adapting helps a lot: in Notebook 01 one epoch of feature
extraction gives ~78 %, three epochs of fine-tuning ~90 %."

**Freezing layers.** "Freezing = `requires_grad = False`. The optimizer ignores these parameters and autograd
does not compute their gradients, which saves compute and memory. Early layers are generic → freeze first; later
layers are task-specific → fine-tune. A typical recipe: first train the head with the backbone frozen, then
unfreeze and fine-tune everything with a small learning rate."
*Pitfall to mention:* freezing weights does **not** freeze BatchNorm running statistics — in `train()` mode BN
keeps updating its mean/variance. Keep frozen parts in `eval()` mode (Notebook 01 and the lab do this).

**Catastrophic forgetting.** "With a too-large learning rate the first gradient steps — driven by a random head —
destroy the pretrained features. Hence: small LR, warm-up (our One-Cycle schedule), or train the head first."

**ImageNet normalisation.** "The pretrained weights expect inputs normalised with ImageNet's per-channel mean
`[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Use exactly the preprocessing the backbone was trained
with, and use exactly the same preprocessing at inference time. In Lab Ex. 3 students will see what a forgotten
`Normalize` does to an INT8 model: accuracy collapses from ~90 % to ~30 %."

**Input size.** "CIFAR-10 images are 32×32; MobileNetV2 reduces spatial size by 32×. We upsample to 96×96 — a
compromise: much better accuracy than 32 px, ~5× cheaper than 224 px."

**Data augmentation.** "Augmentation generates new, plausible training images on the fly: random crops, flips,
colour jitter … It is a regulariser — the network cannot memorise exact pixels. Only on the **training** set;
the test set must represent reality. And augmentations must preserve the label: flipping a cat is fine, flipping
a '6' or a traffic sign with text is not."

### 4.2 Model optimisation (T+1:45, 35 min)

**Why optimise?** "A model that is 90 % accurate but needs 1 s per request, 2 GB of RAM or a GPU may be useless
in production. We optimise for:
* **latency** — user-facing APIs have budgets of tens of milliseconds;
* **throughput & cost** — cloud bills scale with CPU/GPU-seconds; 3× faster = ⅓ of the servers;
* **memory & size** — container images, RAM per worker, download size to phones;
* **edge devices** — phones, Raspberry Pi, microcontrollers: no GPU, limited power, sometimes integer-only hardware;
* **energy** — moving data costs more energy than computing on it; 8-bit data moves 4× less."

**Number formats.**

| Format | Bits (sign/exp/mantissa) | Range | Where it shines |
|---|---|---|---|
| FP32 | 1/8/23 | ±3.4·10³⁸ | default for training and CPU inference |
| FP16 | 1/5/10 | ±65 504 | GPU inference (Tensor Cores); overflow risk in training |
| BF16 | 1/8/7 | same as FP32 | training on TPUs/modern GPUs, CPUs with AMX |
| INT8 | 8-bit integer | −128…127 / 0…255 | CPU inference (VNNI), mobile NPUs, TensorRT |

"Fewer bits → smaller model and less memory traffic. But **speed needs hardware support**: on our laptop CPU,
BF16 is 20–40× *slower* than FP32 because it is emulated (Notebook 02), while INT8 has fast kernels on every
modern x86 and ARM CPU."

**Quantization theory — the affine mapping.**
"We map real values x in a range [x_min, x_max] linearly onto integers q in [q_min, q_max]:"

```
scale       s = (x_max − x_min) / (q_max − q_min)
zero-point  z = q_min − round(x_min / s)                  (an integer)
quantize    q = clamp( round(x / s) + z , q_min, q_max )
dequantize  x̂ = s · (q − z)
```

"The zero-point makes sure that the real value 0.0 maps exactly onto an integer — important because of zero
padding and ReLU. The rounding error is at most s/2, so **the wider the range, the larger the error**: one outlier
can ruin the resolution for all other values. *Symmetric* quantization fixes z = 0 (common for weights),
*asymmetric* uses the full range (common for activations, e.g. after ReLU everything is ≥ 0)."

"A quantized matmul works on integers: `y ≈ s_x·s_w · Σ (q_x − z_x)(q_w)` — INT8 multiplies, INT32 accumulation,
then *requantize* to INT8 with the output scale. That is where the speed comes from."

**Per-tensor vs per-channel.** "Per-tensor: one scale for the whole weight tensor. Per-channel: one scale per
output channel. Channels often have very different magnitudes (Notebook 02 shows a layer where |w|max ranges from
0.004 to 2.7!) — with one shared scale the small channels are rounded to almost nothing. Per-channel weights are
the default in all modern toolchains; activations stay per-tensor."

**Three ways to quantize.**

| | Dynamic quantization | Static PTQ | Quantization-aware training (QAT) |
|---|---|---|---|
| Weights | INT8 ahead of time | INT8 ahead of time | INT8 (learned with fake-quant) |
| Activations | quantized at run time from observed min/max | INT8 with ranges fixed by **calibration** | INT8 with learned/observed ranges |
| Needs data? | no | a few hundred representative samples (no labels) | labelled training data + training run |
| Effort | one line | moderate (fusion, stubs, calibration) | highest |
| Best for | LSTMs, Transformers (matmul-heavy, `nn.Linear`) | CNNs on CPU/edge | when PTQ loses too much accuracy |

"**Calibration** = running representative data through the model with *observers* attached that record the
activation ranges (MinMax, histogram/entropy, percentile). Those ranges become the scales and zero-points.
**QAT** inserts *fake-quantize* operations (quantize → dequantize in float) during fine-tuning so the network
learns to be robust to rounding; gradients pass the non-differentiable rounding with the straight-through
estimator."

**Operator fusion before quantization.** "Conv → BatchNorm → ReLU are fused into one op: BN is folded into
the conv weights (w' = w·γ/σ, b' = (b−μ)·γ/σ + β), ReLU is applied inside the kernel. One read and one write of
the activation instead of three, and only one requantization. Both PyTorch (`fuse_model`) and ONNX Runtime do
this."

**Pruning.** "Many weights are near zero. *Magnitude pruning* sets the smallest |w| to exactly zero.
* **Unstructured** pruning removes individual weights anywhere → high sparsity with little accuracy loss —
  but the tensor keeps its shape. Dense conv/GEMM kernels multiply by zero exactly as fast as by any other
  number, so **latency and dense file size do not change**. Benefits only come with compression (zip), sparse
  formats + sparse kernels, or special hardware (NVIDIA 2:4 structured sparsity).
* **Structured** pruning removes whole filters/channels/heads → the layer really becomes smaller and faster on
  any hardware, but accuracy drops more and you must fix up the next layer (libraries like `torch-pruning`).
* Always prune gradually and fine-tune in between; Notebook 02 shows a 50 % pruned model collapsing and then
  recovering most of its accuracy after one short fine-tuning pass."

**Knowledge distillation (concept only).** "Train a small *student* network to mimic a big *teacher*: the loss
combines the normal cross-entropy with a term that matches the teacher's **softened** output distribution
(softmax with temperature T > 1). The soft targets carry 'dark knowledge' — e.g. that a cat looks more like a dog
than like a truck. DistilBERT is 40 % smaller and 60 % faster than BERT with ~97 % of its quality. Distillation
changes the *architecture*; quantization and pruning compress a *given* architecture — they combine well."

*Loss:* `L = α · CE(y, softmax(z_s)) + (1−α) · T² · KL(softmax(z_t/T) ‖ softmax(z_s/T))`.

**The `torch.ao.quantization` deprecation.** "In PyTorch 2.9 the eager and FX quantization APIs print a
deprecation warning: they move to the separate library **torchao** (`quantize_(model, config)` for eager-style,
and the **PT2E** flow `torch.export → prepare_pt2e → calibrate → convert_pt2e`). The concepts are identical. We
use the eager API because it makes each step visible, and we deploy with ONNX Runtime quantization, which is
independent of this migration."

### 4.3 Export & runtimes (T+3:25, 15 min)

**TorchScript vs ONNX.**
* "**TorchScript** (`torch.jit.trace/script`) serialises a model into a PyTorch-specific IR — needs libtorch to
  run. It is in maintenance mode; the future PyTorch path is `torch.export` (+ AOTInductor / ExecuTorch).
* **ONNX** is an open, framework-neutral graph format (operators + weights, versioned by *opset*). One file runs
  on ONNX Runtime, TensorRT, OpenVINO, CoreML converters, browsers (onnxruntime-web)…
* `torch.onnx.export` traces the model. In torch 2.9 the default is the new `torch.export`-based ("dynamo")
  exporter; we pass `dynamo=False` to use the mature TorchScript-based exporter, which supports `dynamic_axes`
  directly. Either way: name your inputs/outputs, mark the batch dimension dynamic, fix the opset, export in
  `eval()` mode, and **validate numerically** against PyTorch."

**ONNX Runtime.** "ORT is a C++ inference engine with Python/C#/Java/JS bindings. *Execution providers* plug in
hardware: CPU (default, MLAS kernels), CUDA, TensorRT, OpenVINO, CoreML, DirectML… At session creation ORT
optimises the graph:
* **basic**: constant folding, redundant node elimination;
* **extended**: fusions such as Conv+activation (`FusedConv`), GELU/LayerNorm/attention fusion for Transformers;
* **all**: plus layout transformations (NCHWc blocked layout on x86).
For our MobileNetV2 ORT is ~8–10× faster than PyTorch eager at batch 1 — before any quantization — because it runs
a fused, pre-planned graph without Python/dispatcher overhead per op."

**Why we serve ONNX Runtime on Day 3.** "`pip install onnxruntime` is ~15 MB, CPU-only, no CUDA libraries.
`torch` is ~200 MB for the CPU wheel and >2 GB with CUDA. The Day-3 container needs only `onnxruntime`,
`numpy`, `Pillow` and `flask`: smaller image, faster cold starts and scaling, smaller attack surface, and the
same model file can later move to another runtime or language. The `model_card.json` carries preprocessing
parameters and labels, so the API contains no hard-coded model knowledge."

**INT8 in ONNX Runtime.** "`quantize_static` with a `CalibrationDataReader`, QDQ format
(QuantizeLinear/DequantizeLinear pairs that ORT fuses into integer kernels), per-channel weights, U8 activations /
S8 weights on x86. Run `quant_pre_process` first (shape inference + optimisation). `quantize_dynamic` exists but
produces `ConvInteger` ops that are slow for CNNs (NB03 shows ~10× slower than static) — use it for
Transformers/LSTMs, or as a fallback."

---

## 5. Results table — "What difference does it make?"

Students fill in their own column; the reference values were measured on the trainer's laptop
(Intel i7-14650HX, Windows 11, torch 2.9.1+cpu, onnxruntime 1.30; test subset of 2 000 CIFAR-10 images;
latency = median, batch 1, 4 threads). Latencies on a shared laptop are noisy — discuss **ratios**, not
absolute numbers.

| Technique | Size [MB] | Latency [ms] | Accuracy | Δ acc [pp] | Take-away |
|---|---|---|---|---|---|
| PyTorch FP32 baseline | 9.19 | 8.8 | 0.900 | – | reference |
| PyTorch dynamic INT8 (Linear only) | 9.15 | 12.1 | 0.900 | 0.0 | only the 1280→10 head is quantized → no gain for a CNN |
| PyTorch static PTQ INT8 (500 calib. imgs, onednn) | 2.66 | 7.1 | 0.898 | −0.25 | 3.5× smaller, ~1.2× faster, tiny accuracy cost |
| PyTorch QAT INT8 (62 steps) | 2.66 | 7.1 | 0.894 | −0.65 | same size/speed as PTQ; too short to beat PTQ here |
| FP16 weights (storage only) | 4.65 | = FP32 | 0.900 | 0.0 | 2× smaller; speed needs a GPU |
| BF16 on CPU | 4.65 | 227 | 0.901 | +0.1 | ~26× **slower**: no native BF16 units on this CPU |
| Pruned 30 % (global L1, unstructured) | 9.19 (gzip 6.7) | ≈ FP32 | 0.879 | −2.1 | same dense size & speed |
| Pruned 50 % (global L1, unstructured) | 9.19 (gzip 5.2) | ≈ FP32 | 0.331 | −56.9 | collapses without re-training |
| Pruned 50 % + one short fine-tune | 9.19 | ≈ FP32 | 0.874 | −2.7 | fine-tuning recovers most accuracy |
| ONNX Runtime FP32 | 8.92 | 1.3 | 0.900 | 0.0 | ~8× faster than PyTorch eager, same weights |
| **ONNX Runtime INT8 (static QDQ, per-channel)** | **2.61** | **0.9** | **0.894** | −0.6 | **deployed on Day 3**; 96 % top-1 agreement with FP32 |
| ONNX Runtime INT8 (dynamic) | 2.41 | 11.7 | 0.895 | −0.55 | `ConvInteger` path ~13× slower than static |

(Sources: Notebook 02 and Notebook 03 outputs; the model is deterministic, so accuracies reproduce exactly, while
latencies vary run to run — e.g. ORT FP32 vs INT8 at batch 1 ranged from 0.7/0.6 ms to 1.3/1.3 ms depending on
machine load. INT8 wins clearly at larger batches: 2 200 vs 3 400 images/s at batch 128, Lab Ex. 6.)

**Your measurements**

| Technique | Size [MB] | Latency [ms] | Accuracy | Your observation |
|---|---|---|---|---|
| PyTorch FP32 baseline | | | | |
| PyTorch dynamic INT8 | | | | |
| PyTorch static PTQ INT8 | | | | |
| PyTorch QAT INT8 | | | | |
| FP16 weights / BF16 | | | | |
| Pruned 30 % / 50 % / 50 % + fine-tune | | | | |
| ONNX Runtime FP32 | | | | |
| ONNX Runtime INT8 (static) | | | | |

---

## 6. Discussion questions

1. Dynamic quantization made MobileNetV2 neither smaller nor faster. Why? For which model would it be the best
   choice?
2. The INT8 model is ~3.5× — not exactly 4× — smaller than FP32. Where do the extra bytes come from?
3. Why is the zero-point an integer, and why must 0.0 be exactly representable?
4. 50 % of the weights are zero, yet the model is not faster. What would you have to change to get a speed-up?
5. BF16 was *slower* than FP32 on the laptop. On which hardware would it be faster, and why?
6. PyTorch eager vs ONNX Runtime FP32: ~8× difference for the *same* weights. Where does that come from?
7. When would you accept QAT's extra training cost instead of PTQ?
8. Your INT8 model is 0.5 pp less accurate than FP32 but 3.5× smaller. Which one do you ship to (a) a phone app,
   (b) a medical-imaging service? What other metrics would you check (per-class accuracy, agreement rate)?
9. What could make your calibration set unrepresentative in a real deployment?
10. Why is batch-1 latency the relevant number for an interactive API, and throughput for an offline job?

## 7. Common pitfalls (mention them explicitly)

* **Quantized PyTorch models are CPU-only.** Eager INT8 modules don't run on CUDA; for GPUs use FP16/TensorRT.
* **Backend must match the target:** `x86`/`fbgemm` for Intel/AMD servers, `qnnpack` for ARM (phones, Raspberry
  Pi, Apple M-series). The qconfig must match (`fbgemm` uses `reduce_range=True`). Some wheels ship only some
  backends — the Windows torch 2.9.1 CPU wheel only has `onednn`; the notebooks pick the available one
  automatically (`torch.backends.quantized.supported_engines`).
* **Accuracy drop** — always re-evaluate after quantization on the same test set; look at per-class accuracy and
  FP32/INT8 agreement, not just the mean.
* **Calibration data must be representative** and use exactly the inference preprocessing. A forgotten
  `Normalize` in the calibration pipeline silently destroys an INT8 model (Lab Ex. 3: ~90 % → ~30 %). Never
  calibrate on the test set.
* **`model.eval()`** before export, calibration, and evaluation (BatchNorm/Dropout).
* **Frozen ≠ BN frozen** — keep frozen BatchNorm layers in `eval()` mode.
* **Unstructured pruning ≠ speed** — and `prune.remove()` is needed to make pruning permanent before saving.
* **Benchmarking hygiene:** warm-up runs, median of many runs, fixed thread count, same input size, batch 1 *and*
  larger batches, measure on the target hardware, close other heavy programs.
* **ONNX export:** forgetting `dynamic_axes` freezes the batch size to 1; always compare outputs with
  PyTorch (`np.allclose`); keep the file self-contained (no external `.data` file) unless > 2 GB.
* **Deprecations in torch 2.9:** `torch.ao.quantization` → `torchao`; legacy ONNX exporter → `dynamo=True`
  exporter. Both still work in 2.9; pin your versions in `requirements.txt`.
* **Jupyter kernel:** make sure the notebook runs in the course venv (first cell prints the torch version).

## 8. Recap & homework

**Recap (5 sentences for the last slide)**
1. Transfer learning reuses pretrained features — fine-tuning MobileNetV2 gave ~90 % on CIFAR-10 in minutes on a CPU.
2. Quantization maps floats to integers via scale and zero-point; static INT8 with calibration is the go-to for CNNs on CPU.
3. Dynamic quantization is for Linear/LSTM-heavy models; FP16/BF16 need hardware support; QAT buys back accuracy.
4. Unstructured pruning shrinks *compressed* size, not dense latency; structured pruning shrinks the network.
5. ONNX + ONNX Runtime gives a portable, fused, fast model — `model_int8.onnx` + `model_card.json` go to Day 3.

**Homework (choose ≥ 2)**
1. **Knowledge distillation:** train a small CNN (e.g. 3 conv layers, 32×32 input) on CIFAR-10 with and without
   distillation from our fine-tuned MobileNetV2 (T = 4, α = 0.5). Compare accuracy and size.
2. **PT2E / torchao:** `pip install torchao` and quantize the model with the PT2E flow
   (`torch.export.export` → `prepare_pt2e` with `X86InductorQuantizer` → calibrate → `convert_pt2e`). Compare with
   the eager result.
3. **Real structured pruning:** use `torch-pruning` to remove 30 % of the channels, fine-tune 1 epoch, export to
   ONNX and measure size/latency with ONNX Runtime.
4. **Calibration methods:** compare `MinMax`, `Entropy` and `Percentile` in `quantize_static` and per-class accuracy.
5. **Resolution study:** retrain with `--img-size 64` and `--img-size 128` (`scripts/train_and_export.py`) and plot
   accuracy vs latency.
6. Read: Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only
   Inference* (CVPR 2018); Hinton et al., *Distilling the Knowledge in a Neural Network* (2015).

**Preview Day 3:** we wrap `model_int8.onnx` in a Flask REST API (`/predict`, `/health`), with preprocessing in
NumPy/Pillow read from `model_card.json` — no PyTorch in the serving code.
