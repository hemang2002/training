# References: Trainer Reading and Reference Pack

This folder is a **separate activity from the day-by-day course material**. It gives the trainer:
1. what to read **before teaching**, for your own depth, and
2. vetted links to **share with MSc students**.

All links and free-tier facts were **verified on 2026-09-26** with live web checks.

| File | What it is | Use it for |
|---|---|---|
| `reading_list.md` | 152 curated links in 12 topic sections: PyTorch → transfer learning → optimization (quantization / pruning / distillation / ONNX) → detection → GANs → Flask / Streamlit → Docker → Kubernetes → free cloud → MLOps. Each link has type, level, time estimate and why to read it | Your own study; pick the **(S)** items to share with students; ★ marks core items |
| `free_cloud_options.md` | Verified comparison of free platforms for demoing a Dockerized Flask + Streamlit app and a small K8s cluster, with limits, credit-card needs, gotchas and recommendations | Planning the deployment day; student account setup |

---

## How to use this folder

- **Trainer:** follow the must-read top 15 below first, then read the ★ items in whichever section you teach next.
- **Students:** share only the **(S)**-marked links, ideally 2–4 per session, as pre-reading. Don't send the whole list; it overwhelms.
- **Before each run of the course:** re-check `free_cloud_options.md` against the official links. Free tiers changed several times in 2026 alone:
  - Hugging Face Docker Spaces became paid;
  - Oracle A1 was halved;
  - DigitalOcean left the GitHub Student Pack;
  - Play with Kubernetes and Play with Docker shut down.
- **Level tags:** B = beginner, I = intermediate, A = advanced. Time estimates are rough reading times.

---

## Heads-up: things that will trip you up if you rely on older material

1. **PyTorch quantization has moved to `torchao`.** `torch.ao.quantization` (eager/FX) is deprecated. Teach torchao / PT2E, or use **ONNX Runtime quantization**, which is the simplest for the deploy demo.
2. **Hugging Face Spaces:** free accounts can **no longer create Docker (or CPU-Basic Gradio) Spaces**; that now needs PRO ($9/mo). Use **Render** for the no-card Docker demo.
3. **Play with Kubernetes / Play with Docker were discontinued on 1 March 2026.** Use **Killercoda** (1-hour sessions) or local **kind / minikube**.
4. **AWS Free Tier for new accounts** is credit-based: $100 + up to $100, a 6-month Free plan. It's no longer "12 months of t2.micro."
5. **Oracle Always Free A1** is now **2 OCPU / 12 GB**, not the 4 / 24 in most blog posts.

---

## Must-read top 15, before teaching (in this order)

The order follows the course arc. Total is about 12–14 hours, spread over the week before the course. The numbers in brackets are the item numbers in `reading_list.md`.

| # | Read | Why it's on the top list | Time |
|---|---|---|---|
| 1 | Karpathy, "A Recipe for Training Neural Networks" [#7] | The debugging mindset you'll model live in every lab | 25 min |
| 2 | PyTorch "Learn the Basics" [#10] (skim if fluent) | Aligns your code style with the official idioms students will google | 1–2 h |
| 3 | PyTorch Transfer Learning tutorial [#21] + CS231n transfer notes [#22] | The core Day-2 lab and the freeze-vs-fine-tune decision matrix | 1 h |
| 4 | PyTorch quantization page (deprecation) [#31] + torchao docs [#32] | Avoid teaching APIs that are being removed | 1 h |
| 5 | Jacob et al., integer-only quantization [#37] (sections 1–3) | Scale / zero-point math: the conceptual core of quantization | 1 h |
| 6 | ONNX export tutorial [#47] + ORT quantization docs [#52] | The practical path to a small, fast CPU model for free-tier serving | 1.25 h |
| 7 | Pruning tutorial [#41] + Knowledge Distillation tutorial [#44] | The other two optimization labs, including pruning's "sparse isn't automatically faster" caveat | 1.5 h |
| 8 | TorchVision Object Detection Finetuning tutorial [#57] | The core detection lab: target format and replacing the box predictor | 1 h |
| 9 | Lilian Weng, "R-CNN Family" [#61] + Faster R-CNN paper [#65] (RPN section) | Lets you whiteboard RPN, anchors and RoI pooling confidently | 1.5 h |
| 10 | PyTorch DCGAN tutorial [#69] + DCGAN paper [#71] | The core GAN lab and the architecture rules behind it | 1.5 h |
| 11 | Streamlit Caching [#85] + Streamlit Docker [#88] + Flask Gunicorn [#82] | The three most common serving bugs: model reload per rerun, wrong bind address, dev server in prod | 1 h |
| 12 | Docker best practices [#92] + multi-stage builds [#93] | Small, cacheable ML images; your Dockerfile-grading checklist | 50 min |
| 13 | K8s Basics tutorial [#101] + Probes [#110] + HPA walkthrough [#113] | The exact sequence you'll demo; probes and HPA are where students get stuck | 2.5 h |
| 14 | `free_cloud_options.md` (this folder) + Render free docs [#122] | Know the limits (spin-down, 512 MB, no card) before promising students a live URL | 45 min |
| 15 | "Hidden Technical Debt in ML Systems" [#133] + Rules of ML [#134] (skim) | Frames the MLOps session and the "why" of everything above | 1.5 h |

### Next tier (if you have more time)
- Wu et al., NVIDIA integer quantization [#38] and Krishnamoorthi's whitepaper [#39]: deeper quantization practice.
- Hinton distillation [#45] and Lottery Ticket [#43]: great MSc discussion papers.
- Goodfellow's NIPS 2016 GAN tutorial [#72] and Lilian Weng, GAN → WGAN [#74]: instability and mode collapse.
- FPN [#66] and COCO evaluation [#68]: explain `_fpn` and mAP properly.
- Google MLOps maturity levels [#135] and Made With ML [#137]: structure the final session.

---

## Suggested student sharing plan (not day-bound)

| When | Share (S-items) |
|---|---|
| Before the course | 3Blue1Brown [#1], Learn the Basics [#10], Karpathy's Recipe [#7] |
| Around transfer learning | Transfer Learning tutorial [#21], CS231n transfer notes [#22], TorchVision models [#24] |
| Around optimization | Practical Quantization blog [#36], Pruning [#41], Distillation [#44], ONNX export [#47], Netron [#54] |
| Around detection / GANs | Detection finetuning [#57], Lilian Weng R-CNN [#61], COCO eval [#68], DCGAN tutorial [#69], Google GAN course [#73], ganhacks [#75] |
| Around serving / Docker | Flask Quickstart [#79], Streamlit Get Started [#83] + Caching [#85], Docker Get Started [#90], Compose Quickstart [#98] |
| Around Kubernetes | K8s Basics [#101], minikube start [#102] (install beforehand), kind [#105], kubectl cheat-sheet [#118], Killercoda [#131] |
| Around cloud / MLOps | Azure for Students [#129], Render free [#122], Hidden Tech Debt [#133], Rules of ML [#134], Made With ML [#137], MLOps Zoomcamp [#146] |

---

## Verification notes
- Every URL in `reading_list.md` returned HTTP 200 on 2026-09-26. Redirecting URLs were replaced with their final destination.
- Removed as dead or moved:
  - the PyTorch Flask REST API tutorial;
  - an old torchao blog URL (404);
  - one Medium article that blocks automated access.
- In `free_cloud_options.md`, facts that couldn't be confirmed on an official page are marked **"verify before class"**.
