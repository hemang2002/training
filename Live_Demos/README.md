# Live Demos — Object Detection & GANs, trained **and optimised** on a laptop CPU

Four notebooks for live demonstrations between the regular course days of
*"Train → Optimize → Serve → Containerize → Deploy ML models with PyTorch"*.
Everything runs on a CPU-only laptop, in front of students, with visible progress and visual results.

| Notebook | What it shows | Measured run time (reference laptop, see below) |
|---|---|---|
| `01_object_detection_pennfudan.ipynb` | Fine-tune COCO-pretrained Faster R-CNN (MobileNetV3-Large-FPN) on Penn-Fudan pedestrians; own AP@0.5 implementation; before/after comparison | live training **11.0 min** (6 epochs, ~65 s/epoch + eval); whole notebook ~1.5 min with `USE_CACHED=True` |
| `02_detection_optimization.ipynb` | Resolution, proposals, dynamic INT8, threads, ONNX Runtime — latency **and** AP for each | **~4–4.5 min** (14 variants × latency + AP, 2 ONNX exports) |
| `03_dcgan_fashionmnist.ipynb` | DCGAN from scratch on FashionMNIST, per-epoch sample grids, GIF, latent interpolation | live training 10 epochs: **18.6 min** measured while other jobs shared the CPU (76–157 s/epoch) — expect ~10–13 min on an idle laptop; `TIME_BUDGET_MIN=15` stops early if needed; ~20 s cached |
| `04_gan_optimization.ipynb` | Dynamic-quant trap, Conv-BN fusion, FP16/BF16, ONNX Runtime, ORT static INT8, pruning, distillation | **~2 min** cached (distillation training itself: 178 s when `USE_CACHED=False`) |
| `prepare_demo.py` | Night-before download of datasets + pretrained weights | ~1–2 min (network) |

**Reference laptop:** Windows 11, 24 logical CPU cores, **CPU only** (no GPU used), torch 2.9.1, torchvision 0.24.1, onnxruntime 1.30.
All numbers below come from the executed notebooks (`outputs/*_results.csv`). Latency = best-of-3-trials median at batch 1.

### Detection (Penn-Fudan, 50 held-out images)

| Model | AP@0.5 | AP@[.5:.95] | Precision @ score ≥ 0.5 | False positives |
|---|---|---|---|---|
| COCO-pretrained, "person" class only (no fine-tuning) | 0.984 | 0.804 | 0.611 | 72 |
| new 2-class head, untrained | 0.023 | 0.006 | 0.022 | 4222 |
| **fine-tuned (6 epochs, 120 images, 11 min CPU)** | **0.976** | **0.793** | **0.915** | **10** |

(Our AP implementation matches `pycocotools` within 0.003.) The COCO model already finds every pedestrian (recall 1.0)
but also boxes people the dataset does not label (far away, cut off at the border). Fine-tuning teaches the
**annotation convention** → precision 0.61 → 0.92. See `outputs/det_predictions_before_after.png`.

| Optimisation (detector) | Size MB | Latency ms | Speed-up | AP@0.5 |
|---|---|---|---|---|
| baseline FP32 eager, 800 px, 1000 proposals, 16 threads | 76.0 | 121 | 1.00× | 0.976 |
| input 640 / 512 / 416 / 320 px | 76.0 | 86 / 60 / 46 / 39 | 1.4× / 2.0× / 2.7× / 3.1× | 0.979 / 0.951 / 0.934 / 0.855 |
| dynamic INT8 (Linear layers only) | **34.3** | 116 | 1.04× | 0.975 |
| 300 / 100 / 50 proposals | 76.0 | 100 / 80 / 76 | 1.2× / 1.5× / 1.6× | 0.976 / 0.968 / 0.935 |
| 1 thread (vs 16) | 76.0 | 353 | 0.34× | 0.976 |
| ONNX Runtime FP32 | 76.0 | 59 | 2.1× | 0.983 |
| **512 px + 100 proposals, ONNX Runtime** | 76.0 | **25** | **4.9×** | 0.944 |

### GAN (DCGAN generator, 1.78 M params)

| Variant | Size MB | ms @ batch 1 | ms @ batch 64 | PSNR vs FP32 |
|---|---|---|---|---|
| FP32 eager (baseline) | 7.13 | 0.86 | 8.0 | — |
| FP16 / BF16 eager (CPU) | 3.57 | 25.2 / 22.9 | 183 / 119 | 75 / 57 dB |
| ONNX Runtime FP32 | 7.13 | **0.32** | 18.0 | 144 dB (identical) |
| ONNX Runtime INT8 static (QDQ) | **1.79** | 0.92 | 21.7 | 30 dB |
| pruned 30 / 50 / 70 / 90 % (L1 unstructured) | 7.13 (gzip 5.1 / 4.0 / 2.7 / 1.2) | 0.87–0.93 | 8.6–8.8 | 28 / 18 / 12 / 10 dB |
| **distilled student (ngf=16, 0.35 M params)** | **1.39** | **0.50** | **1.5** | 23 dB |

GAN training: shapes appear after epoch 1–2, clearly recognisable shirts/trousers/sneakers/bags by epoch 5
(`outputs/gan_progress_strip.png`, `outputs/gan_progress.gif`).

---

## 1. When to run them

| Demo | Suggested slot | Why there |
|---|---|---|
| **GAN demo** (03 → 04) | End of **Day 1** (after the training-loop session) or end of **Day 2** | Day 1 students just wrote their own training loop — a GAN is "two training loops fighting". 04 previews Day-2 optimisation vocabulary (quantization, ONNX, pruning, distillation). |
| **Detection demo** (01 → 02) | **Day 2**, right after *transfer learning*; run 02 right after the *quantization* session | 01 is transfer learning on a harder task (boxes, not labels). 02 shows the quantization session's tools on a real model — and that on a detector the *biggest* wins come from other knobs. |

Each demo (training notebook + optimisation notebook) is **~60–75 min including explanation**:

* ~10 min: concepts (talking points below) — while the training cell runs.
* ~10–12 min: live training (or instant with the cached checkpoint).
* ~10 min: evaluation / visual results, discussion.
* ~20–25 min: optimisation notebook, discussion of the final table.
* ~10 min: Q&A, alternatives for bigger projects.

## 2. Pre-demo checklist (do this the night before!)

1. **Download everything** (needs internet, ~200 MB):
   ```bash
   cd Live_Demos
   ../.venv/Scripts/python prepare_demo.py        # Windows;  ../.venv/bin/python on Linux/macOS
   ```
   It downloads Penn-Fudan (`../data/PennFudanPed`, 170 images), FashionMNIST (`../data/FashionMNIST`) and the
   COCO-pretrained detector weights into the torch hub cache (`~/.cache/torch/hub/checkpoints`). It is idempotent.
   * Penn-Fudan URL: `https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip` (verified working). That server does
     not send its intermediate TLS certificate, so plain Python fails with `CERTIFICATE_VERIFY_FAILED` on some machines
     — the script uses the `certifi` CA bundle, which fixes it. Fallback mirror (verified identical file, 53.7 MB):
     `https://web.archive.org/web/2023id_/https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip`.
2. **Run all four notebooks once, in order (01 → 02 → 03 → 04)**. This produces the safety-net files — **keep them**:
   * `checkpoints/fasterrcnn_mbv3_pennfudan.pth` (fine-tuned detector + training history)
   * `checkpoints/dcgan_fashionmnist_G.pth` (generator + discriminator + history + per-epoch sample grids)
   * `checkpoints/dcgan_student_ngf16.pth` (distilled student generator)
   * `checkpoints/*.onnx` (re-created by notebooks 02/04 every run)
   * `outputs/*.png`, `outputs/gan_progress.gif`, `outputs/*_results.csv` — good for slides.
3. **`USE_CACHED` flag** (first code cell of 01, 03, 04): `True` (default) loads an existing checkpoint instead of
   training, so the demo *cannot* fail live. To train live, set `USE_CACHED = False` — the new checkpoint overwrites
   the old one only when training finishes. **Tip:** copy `checkpoints/` to a USB stick as a second backup.
4. Before going on stage: plug in the laptop (battery mode throttles the CPU 2–3×), close browsers/IDEs indexing in the
   background, set Windows power mode to "Best performance". Timings in notebook 02/04 are sensitive to background load.
5. Open the notebooks in Jupyter, *Kernel → Restart & Clear Output*, zoom the browser to ~125 % for the projector.

## 3. Talking points — Object detection

**Task.** Classification says *what*; detection says *what + where* for a variable number of objects.
Output per object: box `[x1, y1, x2, y2]`, class label, confidence score.

**Bounding boxes & IoU.** IoU = area(A∩B)/area(A∪B) ∈ [0,1]. Used (1) to decide if a prediction matches a ground truth
(IoU ≥ 0.5 in PASCAL VOC / AP50), (2) inside NMS, (3) to assign anchors to ground truth during training.

**Anchors.** Pre-defined reference boxes at several scales and aspect ratios tiled over every feature-map position.
The network does not predict boxes from nothing — it predicts *"is there an object in this anchor?"* and *"how to shift/
scale this anchor"* (box deltas). Anchor-free detectors (FCOS, CenterNet, YOLOv8+) predict centres/distances instead.

**Two-stage vs one-stage.**
* *Two-stage* (R-CNN family): stage 1 proposes regions (RPN), stage 2 classifies & refines each region. Accurate, slower.
* *One-stage* (SSD, RetinaNet, YOLO): predict class + box densely in one pass. Faster, historically less accurate on
  small objects (RetinaNet's focal loss closed most of the gap).

**Faster R-CNN architecture** (what notebook 01 fine-tunes):
```
                 ┌──────────────────────────── Faster R-CNN ─────────────────────────────┐
 image ─► resize │  Backbone        FPN (P2..P6)       RPN                RoI heads       │
 (short  +norm ─►│ MobileNetV3 ─►  multi-scale   ─►  anchors ─► objectness   RoIAlign 7x7  │
  side 800)      │  (conv feats)   feature maps  │    + box deltas ─► NMS ─► ~1000 props ─┼─► fc6 ─► fc7 ─┬─► class scores (C)
                 │                               └──────────────────────────────────────►│               └─► box deltas (4C)
                 └──────────────────────────────────────────────────────────────────────┘         ─► score thr + NMS ─► detections
 Transfer learning: keep everything, replace only the last "FastRCNNPredictor" (91 COCO classes → 2: background, person)
```

**FPN (Feature Pyramid Network).** Deep feature maps are semantically strong but coarse; shallow ones are fine but weak.
FPN adds a top-down path with lateral connections so *every* scale gets strong features → small and large objects.

**NMS (Non-Maximum Suppression).** Detectors fire many overlapping boxes per object. Sort by score, keep the best, drop
all others with IoU > threshold (0.5–0.7) to it, repeat. Used twice in Faster R-CNN (on proposals, on final boxes).

**mAP.** For each class: sort detections by confidence, mark TP/FP by IoU matching (each GT matched once; duplicates are
FP), build the precision–recall curve, AP = area under it. mAP = mean over classes. COCO mAP additionally averages over
IoU thresholds 0.50:0.05:0.95 (rewards tight boxes). Notebook 01 implements AP@0.5 from scratch and cross-checks it
against `pycocotools`.

**Discussion prompt.** The COCO model already knows "person" — why does fine-tuning still help? (Annotation
conventions: Penn-Fudan ignores many background/occluded people; box style; domain; and you get a specialised,
simpler 2-class head.)

## 4. Talking points — GANs

**Generator vs discriminator (minimax).** G maps noise z ~ N(0, I) to images; D outputs P(real).
`min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]`. D is trained as a binary classifier (real=1, fake=0); G is trained
to fool D. In practice G uses the *non-saturating* loss `−log D(G(z))` (strong gradients when D confidently rejects
fakes early in training). G never sees a real image — it learns only through D's gradients.

**DCGAN guidelines (Radford et al. 2015):** strided convs instead of pooling; transposed convs in G; BatchNorm in both
(not on G output / D input); no FC hidden layers; ReLU in G + tanh output; LeakyReLU(0.2) in D; init N(0, 0.02);
Adam lr 2e-4, β₁ = 0.5.

**Mode collapse.** G finds a few outputs that fool D and produces only those (e.g. only trousers). D catches up, G
jumps to another mode, … Spot it by looking at sample *diversity* (the fixed-noise grid in notebook 03).
Remedies: minibatch discrimination, unrolled GANs, WGAN-GP, spectral norm, more data augmentation (DiffAugment/ADA).

**Training instability.** Two players optimising different objectives = no guaranteed convergence; oscillation is
normal. If D becomes perfect, G's gradients vanish; if G's lr is too high, it overshoots. Tricks: β₁ = 0.5, label
smoothing, TTUR (different lrs), spectral normalisation, gradient penalty.

**Why the losses don't indicate quality.** Each loss is measured against a *moving* opponent. A lower G loss may just mean
D got worse. There is no "validation accuracy". Use sample inspection and metrics like FID (distance between Inception
feature statistics of real vs generated images), precision/recall for generative models.

**Context:** diffusion models now dominate high-fidelity generation, but GANs sample in *one forward pass* (vs tens to
hundreds for diffusion) — which is exactly why GAN generators are attractive to optimise and deploy (notebook 04).

## 5. "What difference does optimisation make?" — discussion guide

Put the two final tables on the projector and ask the class to fill in **"what did it buy, what did it cost?"**:

1. **The biggest detection win was not quantization.** Input resolution (3.1× at 320 px) and fewer proposals
   (1.6×) need no retraining at all; ONNX Runtime (2.1×) needs only an export. Stacked: **4.9× faster for −0.03 AP**.
   Ask: *why does 320 px hurt AP so much more than 512 px?* (small/far pedestrians shrink below the receptive field).
2. **Dynamic INT8 shrank the detector 76 → 34 MB but gave ~no speed-up** — it only quantizes `nn.Linear` (the box
   head, where fc6 holds 12.8 M of 19 M params), while time is spent in convolutions. Size ≠ speed.
3. **FP16/BF16 halve the size but are ~25× slower on this CPU** — no native half-precision math, so every op
   converts. On a GPU the same change is typically ~2× faster. *Always measure on the deployment hardware.*
4. **INT8 static on the GAN: 4× smaller, not faster at this tiny size, and visibly lower quality (PSNR 30 dB)** —
   calibrated on random latents only. Discuss: what calibration data would a generator need?
5. **Unstructured pruning never made anything faster** — zeros in a dense tensor still get multiplied. It only helps
   compression (gzip) or sparse-aware hardware/kernels. Quality collapses beyond ~50 %.
6. **Distillation was the only technique that gave smaller *and* faster** (5× fewer params, 1.7× at batch 1,
   5.3× at batch 64) — because it changes the architecture — at the cost of a training run and some quality.
7. **Threads:** 1 thread was 2.9× slower than 16, but in a server you'd often run several processes with fewer
   threads each for better throughput — connects to gunicorn workers / Kubernetes replicas on Day 4.
8. Meta-lesson: **latency numbers are fragile** — we had to warm up on every input shape, warm up ≥ 1 s, and take the
   best of 3 trials to get stable numbers on a laptop. Ask students how they'd benchmark in production (p95 under real load).

## 6. Going bigger (alternatives to mention)

* **Ultralytics YOLO (v8/v11)** — `pip install ultralytics`; state-of-the-art one-stage detector, trains Penn-Fudan-sized
  data in minutes on a GPU, one-line export to ONNX/OpenVINO/TensorRT/CoreML. **License: AGPL-3.0** — using it in a
  closed-source product/service requires open-sourcing your code *or* buying an Ultralytics enterprise license.
  torchvision models (BSD-3) have no such constraint.
* **Mask R-CNN** (`torchvision.models.detection.maskrcnn_resnet50_fpn_v2`) — adds an instance-segmentation mask head;
  Penn-Fudan already ships masks, so notebook 01's dataset class needs only to return `masks` too (the official
  torchvision tutorial does exactly this). Too slow for a live CPU demo (ResNet-50), fine on a GPU.
* **Larger detectors:** `fasterrcnn_resnet50_fpn_v2` (COCO mAP 46.7), RetinaNet, FCOS, DETR/RT-DETR (transformers).
* **Larger GANs:** DCGAN on CelebA 64×64, conditional GAN (class-conditioned FashionMNIST), StyleGAN2-ADA; or a small
  diffusion model (DDPM) for comparison.
* **Google Colab (free T4 GPU)** — the same notebooks run unchanged (they auto-use CPU; add `.to("cuda")` for GPU):
  detection training drops from ~10 min to < 1 min, allowing ResNet-50 backbones and more epochs.
* **Further optimisation paths:** static INT8 of the conv backbone via ONNX Runtime / OpenVINO / `torchao` PT2E,
  TensorRT on NVIDIA GPUs, structured pruning + fine-tuning, `torch.compile`.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `AssertionError: Run prepare_demo.py first` | Run `prepare_demo.py` (needs internet) |
| `CERTIFICATE_VERIFY_FAILED` downloading Penn-Fudan | Already handled via `certifi`; otherwise download the mirror URL above manually and unzip into `../data` |
| Training much slower than the table | Laptop on battery / background load; use `USE_CACHED=True` |
| `DeprecationWarning: torch.ao.quantization ...` | Expected in PyTorch 2.9 (API removed in 2.10, successor `torchao`); notebooks silence it |
| ONNX export prints TracerWarnings | Expected for detection models with the TorchScript exporter; notebooks verify ORT outputs match PyTorch |
| Notebook 02/04 says checkpoint missing | Run 01 / 03 first |
