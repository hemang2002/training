# Day 1 — PyTorch Fundamentals: from Tensors to a Trained CNN

> **Course:** Train → Optimize → Serve → Containerize → Deploy ML models with PyTorch (5 days)
> **Audience:** MSc students (comfortable with Python and NumPy; basic ML theory helpful but not required)
> **Duration:** 5 hours including 2 breaks. About 35% is dedicated student lab time, plus follow-along live coding.

> 🆕 **6-hour version:** the timetable in [`../Day1_6h_Plan/README.md`](../Day1_6h_Plan/README.md) (Check → Show → Solve → Conclude, with check-in and exit quizzes) replaces the one below. The concept notes here are still the background for it.

This README is the **trainer's lecture guide**. Each block has talking points you can speak from, ASCII diagrams you can draw on the whiteboard, common student mistakes, and discussion questions.

---

## 📁 Materials in this folder

| File | Purpose | Used in block |
|---|---|---|
| `01_tensors_and_autograd.ipynb` | Tensors, broadcasting, matmul, devices, autograd, gradient descent from scratch → `nn.Linear` + `optim`. Ends with **Lab 1** exercises (with solutions). | B2, B3, B4 (Lab 1) |
| `02_train_cnn_fashionmnist.ipynb` | Full pipeline on FashionMNIST: data, MLP vs CNN, training/eval loop, curves, confusion matrix, misclassified images, save/reload `state_dict`, comparison table. | B6 (live coding) |
| `03_student_lab.ipynb` | **Lab 2**: 6 exercises + bonus (optimizers, BatchNorm/Dropout, early stopping, augmentation, custom Dataset, LR schedulers, checkpoint/resume). Each has a hint, a TODO cell and a solution. | B9 (Lab 2) |
| `models/` | Created by notebook 02: `fashion_cnn.pth` (best CNN `state_dict`), `fashion_mlp.pth`; lab bonus writes `lab_checkpoint.pt`. | — |

Datasets download automatically into the shared course folder `../data` (FashionMNIST ≈ 30 MB).

### Before the session (trainer checklist)
- [ ] Run all three notebooks once on the presentation machine. This also pre-downloads FashionMNIST.
- [ ] Ask students to install the environment **before** Day 1 (`torch`, `torchvision`, `matplotlib`, `scikit-learn`, `pandas`, `jupyter`) and run the first cell of notebook 01.
- [ ] If Wi-Fi is weak, bring `data/FashionMNIST` on a USB stick.
- [ ] Notebook 02 takes about **3–5 min** end-to-end on a laptop CPU (`QUICK_RUN=False`). Start its training cells **before** you begin explaining the curves, or set `QUICK_RUN=True` for a 30-second demo.

---

## 🎯 Learning objectives

By the end of Day 1 students can:

1. Create and manipulate tensors and reason about **shape, dtype and device**, including broadcasting and matrix multiplication.
2. Explain the **computational graph** and **autograd**, and use `requires_grad`, `backward()`, `.grad`, `no_grad()` and `detach()` correctly.
3. Implement gradient descent **from scratch**, then map every manual step to `nn.Module`, loss functions and `torch.optim`.
4. Build a data pipeline with `Dataset`, `transforms` and `DataLoader`, using a proper **train/validation/test** split.
5. Write a reusable **training loop** and **evaluation loop** (`model.train()` / `model.eval()`).
6. Diagnose **overfitting and underfitting** from learning curves, and apply remedies (dropout, BatchNorm, augmentation, early stopping, LR schedules).
7. **Save and reload** models via `state_dict`, and make experiments **reproducible** with seeds.

---

## 🗓️ Timetable (5 h)

Times are relative to the start (e.g. 0:00 = 09:30).

| Time | Block | Type | Material |
|---|---|---|---|
| 0:00 – 0:15 | **B1** Welcome, course roadmap, environment check | Lecture | README §B1 |
| 0:15 – 0:40 | **B2** Tensors: shape, dtype, device, broadcasting, matmul | Lecture + live coding | NB01 Part 1 |
| 0:40 – 1:05 | **B3** Autograd and gradient descent from scratch → `nn` + `optim` | Lecture + live coding | NB01 Parts 2–4 |
| 1:05 – 1:30 | **B4 🧪 STUDENT LAB 1**: tensor & autograd exercises | **Hands-on** | NB01 Part 5 |
| 1:30 – 1:40 | ☕ **Break 1** | | |
| 1:40 – 2:05 | **B5** Building blocks: `nn.Module`, losses, optimizers, `Dataset`/`DataLoader`, the training loop | Lecture | README §B5 |
| 2:05 – 2:45 | **B6** Full pipeline on FashionMNIST (students follow along) | Live coding | NB02 |
| 2:45 – 3:00 | **B7** Generalisation: overfitting, regularisation, checkpoints, devices, reproducibility | Lecture | README §B7 |
| 3:00 – 3:10 | ☕ **Break 2** | | |
| 3:10 – 4:30 | **B9 🧪 STUDENT LAB 2**: experiments (6 exercises + bonus) | **Hands-on** | NB03 |
| 4:30 – 4:45 | **B10** Lab debrief: students share results, walk through tricky solutions | Discussion | NB03 solutions |
| 4:45 – 5:00 | **B11** Recap, homework, teaser for Day 2 | Lecture | README §B11 |

**Time budget:** labs 105 min (35%), live coding 40 min (students type along, so effectively hands-on too), lecture about 110 min, debrief/discussion 15 min, breaks 20 min.

> **If you run late:** in Lab 2, make Exercises 1, 2, 3 mandatory and 4, 5, 6 optional/homework. In B6, set `QUICK_RUN=True`.
> **If you run early:** let students attempt the Lab 2 bonus, or push the CNN in NB02 to 10 epochs and compare.

---

## B1 · Welcome & roadmap (0:00 – 0:15)

**Talking points**
- The week follows the life-cycle of a real ML model:

```
 Day 1            Day 2                 Day 3            Day 4              Day 5
 TRAIN   ──────►  OPTIMIZE   ─────────► SERVE  ────────► CONTAINERIZE ────► DEPLOY
 PyTorch basics   transfer learning     model as an API  Docker image       cloud / cluster
 CNN from scratch + quantization
```
- Today is the foundation. Everything later (fine-tuning, quantization, serving) assumes you understand tensors, `nn.Module`, `state_dict` and the training loop.
- Why PyTorch? Pythonic, eager execution (debug with `print`), dominant in research, and a mature production story (TorchScript/`torch.export`, ONNX, TorchServe, quantization).
- **Environment check:** everyone runs the first code cell of NB01. Fix installation problems now, not during the lab.

**Discussion question:** *"Who has trained a model in scikit-learn? What does `.fit()` hide from you?"* (Answer: the optimisation loop. Today we open that box.)

---

## B2 · Tensors (0:15 – 0:40) — NB01 Part 1

### Concept: what is a tensor?
- An n-dimensional array, like NumPy's `ndarray`, with two extras: it can live on an **accelerator** (GPU/MPS) and it can **record operations for autograd**.
- Every tensor has three attributes to check first when debugging: **shape, dtype, device**.

```
 rank 0       rank 1          rank 2               rank 4 (images, PyTorch = NCHW)
  5.0       [1, 2, 3]      [[1, 2, 3],          (batch, channels, height, width)
 shape ()   shape (3,)      [4, 5, 6]]          e.g. (64, 1, 28, 28)
                            shape (2, 3)
```

### Talking points
- **Creation:** `torch.tensor`, `zeros`, `ones`, `arange`, `rand`, `randn`. `from_numpy` **shares memory**; `torch.tensor(arr)` **copies**.
- **dtype:** floats default to `float32` (the DL standard). Ints default to `int64` (class labels). NumPy defaults to `float64`, so call `.float()` when converting. Lower precision (`float16`, `int8`) saves memory: that is **Day 2 quantization**.
- **Reshaping:** `reshape` (safe), `view` (no copy, needs contiguous memory), `unsqueeze`/`squeeze`, `permute` (CHW → HWC for matplotlib), `flatten(1)` (conv maps → linear layer).
- **Reductions and `dim`:** "the `dim` you pass is the dimension that disappears." `argmax(dim=1)` on logits `(B, C)` gives the predicted class per sample.
- **Broadcasting rules**, comparing shapes from the right:

```
  (64, 784)   batch of images           (3, 1)
- (    784)   per-pixel mean          + (1, 4)
= (64, 784)                           = (3, 4)
```
  1. Equal dimensions are fine. 2. A size-1 dimension is stretched. 3. Missing dimensions are added on the left. 4. Anything else is an error.
- **Matmul:** `(n, k) @ (k, m) → (n, m)`. A linear layer is `x @ W.T + b`. `*` is element-wise, `@` is matrix product.
- **Devices:** `device = "cuda" if torch.cuda.is_available() else "cpu"`. `tensor.to(device)` returns a **new** tensor (re-assign it!), while `module.to(device)` moves parameters in place.

### ⚠️ Common mistakes
| Mistake | Symptom | Fix |
|---|---|---|
| `(N,1)` prediction vs `(N,)` target | **No error**, loss silently wrong (broadcasts to `(N,N)`) | `squeeze(1)` / check shapes |
| `x.to(device)` without re-assigning | "Expected all tensors to be on the same device" | `x = x.to(device)` |
| `float64` data from NumPy into a model | "expected scalar type Float but found Double" | `.float()` |
| `*` instead of `@` | shape error, or wrong result silently | know the difference |
| `.numpy()` on a GPU / grad tensor | RuntimeError | `.detach().cpu().numpy()` |

### Discussion questions
- What shape does `torch.zeros(8,1,5) + torch.zeros(3,1)` produce, and why?
- Why does PyTorch default to `float32` and not `float64` for deep learning?
- Why is memory sharing between NumPy and PyTorch both useful and dangerous?

---

## B3 · Autograd & gradient descent (0:40 – 1:05) — NB01 Parts 2–4

### Concept: the computational graph
When a leaf tensor has `requires_grad=True`, every operation on it is recorded in a **directed acyclic graph**. `loss.backward()` walks the graph in reverse and applies the **chain rule**, storing ∂loss/∂leaf in each leaf's `.grad`.

```
          FORWARD (build graph, compute values)  ─────────────►
  w ──┐
      ├──[ mul ]──► w·x ──┐
  x ──┘                   ├──[ add ]──► ŷ ──[ − y ]──[ pow 2 ]──[ mean ]──► L
  b ──────────────────────┘
          ◄─────────────  BACKWARD (chain rule, fill .grad)
  ∂L/∂w = mean( 2(ŷ − y) · x )        ∂L/∂b = mean( 2(ŷ − y) )
```

- The graph is **dynamic** (define-by-run): rebuilt on every forward pass, so Python `if`/`for` in the model just work.
- **Leaves** (parameters) keep `.grad`. Intermediates don't, to save memory.
- After `backward()` the graph is **freed** (call it twice and you get an error, unless `retain_graph=True`).

### Talking points
1. `y = x**2 + 3x`, then `y.backward()` gives `x.grad = 7` at x=2. Check it on the board with calculus.
2. **Gradients accumulate:** calling `backward()` 3 times gives 7, 14, 21. That is why every loop has `optimizer.zero_grad()`. (Accumulation is useful on purpose for large effective batch sizes.)
3. **Turning autograd off:** `with torch.no_grad():` for evaluation and manual updates, `torch.inference_mode()` for even faster inference, and `.detach()` to pull values out for logging/plotting.
4. **Gradient descent from scratch** on y = 3x + 2 + noise. The five steps:

```
   ┌────────────────────────────────────────────────────────┐
   │ 1. forward     ŷ = w·x + b                             │
   │ 2. loss        L = mean((ŷ − y)²)                      │
   │ 3. backward    L.backward()   → w.grad, b.grad         │
   │ 4. update      w ← w − η·w.grad   (inside no_grad)     │
   │ 5. zero grads  w.grad.zero_()                          │
   └──────────────── repeat for each epoch ─────────────────┘
```
5. The **same** thing with library pieces. Show this mapping table; it is the key "aha" of the morning:

| Manual | PyTorch |
|---|---|
| `w`, `b` with `requires_grad=True` | `nn.Linear(1, 1)` |
| `((ŷ − y)**2).mean()` | `nn.MSELoss()` |
| `w -= lr * w.grad` | `optimizer.step()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |

6. **Learning rate:** too small means slow, too large means divergence (students see it in Lab 1, Ex 1.3). Draw a parabola with steps that overshoot.

### Optimizers in one slide
- **SGD:** θ ← θ − η∇L. Simple, and with a well-tuned LR and schedule often the best at generalising.
- **SGD + momentum:** keeps a velocity v ← βv + ∇L, θ ← θ − ηv. It speeds through ravines and damps oscillation. Think of a ball rolling downhill.
- **Adam:** per-parameter adaptive step sizes from running averages of the gradient (1st moment) and squared gradient (2nd moment), with bias correction. Robust default, `lr=1e-3`. **AdamW** decouples weight decay (the preferred variant for transformers and fine-tuning).
- Each optimizer has its own sensible LR range: SGD ≈ 0.01–0.1, Adam ≈ 1e-4–3e-3.

### ⚠️ Common mistakes
- Forgetting `zero_grad()`: gradients grow each step and training becomes erratic.
- Updating parameters **outside** `no_grad()`: "a leaf Variable that requires grad is being used in an in-place operation".
- Calling `backward()` on a non-scalar: "grad can be implicitly created only for scalar outputs". Reduce with `.mean()` first.
- Using `loss` (a tensor) in a Python list for logging keeps the whole graph alive (memory leak). Use `loss.item()`.

### Discussion questions
- Why does PyTorch accumulate gradients instead of overwriting them?
- Our learned w is 2.91, not 3.00. Is that a bug? (No: it is the least-squares optimum for *this noisy sample*.)
- What would happen with `lr=1.0` in our line fit? Can you predict the threshold? (≈ 2 / curvature)

---

## B4 · 🧪 STUDENT LAB 1 (1:05 – 1:30) — NB01 Part 5

| Exercise | Time | Skill |
|---|---|---|
| 1.1 Shape prediction (broadcasting, matmul, permute) | 5 min | shapes |
| 1.2 Verify autograd against calculus | 5 min | autograd |
| 1.3 Make gradient descent diverge (large LR) | 5 min | learning rate |
| 1.4 Fit a parabola from scratch (+ bonus with `nn.Linear` + Adam) | 10 min | full loop |

**Trainer tips:** walk around. The most frequent issues are a forgotten `no_grad()` around the update, a forgotten `zero_()`, and reusing variables from earlier cells. Solutions are in the notebook directly under each exercise, so ask students to try before revealing.

---

## ☕ Break 1 (1:30 – 1:40)

---

## B5 · Building blocks of a training pipeline (1:40 – 2:05)

### The big picture
```
 ┌─────────┐  indexable   ┌────────────┐  mini-batches  ┌──────────────┐ logits ┌────────┐
 │ Dataset │ ───────────► │ DataLoader │ ─────────────► │ Model        │ ─────► │  Loss  │
 │ (x, y)  │  __getitem__ │ batch,     │  (B,C,H,W),(B,)│ (nn.Module)  │        │        │
 └─────────┘  __len__     │ shuffle    │                └──────▲───────┘        └───┬────┘
      ▲                   └────────────┘                       │ step()             │ backward()
  transforms                                            ┌──────┴──────┐              │
 (ToTensor, Normalize,                                  │  Optimizer  │ ◄──── grads ─┘
  augmentation)                                         └─────────────┘
```

### `Dataset` and `DataLoader`
- A (map-style) `Dataset` implements `__len__` and `__getitem__(i) → (x, y)`. `torchvision.datasets` provides common ones; for your own data you write the class yourself (Lab 2, Ex 5).
- **Transforms** run per sample at load time: `ToTensor()` (PIL uint8 H×W → float C×H×W in [0,1]) and `Normalize(mean, std)`. Compute mean/std on the **training** set only.
- `DataLoader` handles batching, **shuffling** (train only), and parallel loading (`num_workers`; keep it 0 in Jupyter on Windows/macOS).
- Batch size trade-off: larger batches give more stable gradients and better hardware utilisation but more memory, and may need a larger LR.

### `nn.Module`
- `__init__` creates layers, whose parameters are registered automatically (visible in `model.parameters()` / `state_dict()`). `forward` defines the computation.
- Call `model(x)`, **not** `model.forward(x)` (hooks).
- Containers: `nn.Sequential`, `nn.ModuleList`. A plain Python list of layers is **not** registered, which is a classic bug.
- MLP vs CNN. A CNN uses **local receptive fields**, **weight sharing** and **pooling**. It needs fewer parameters and has a better inductive bias for images.

```
 Conv2d 3×3 over an image (weight sharing: the same 9 weights at every position)
 ┌───┬───┬───┬───┐
 │ ▓ │ ▓ │ ▓ │   │       output(i,j) = Σ kernel · patch(i,j) + bias
 ├───┼───┼───┼───┤       out_size = (in + 2·pad − k)/stride + 1
 │ ▓ │ ▓ │ ▓ │   │       28×28, k=3, pad=1 → 28×28 ; MaxPool2d(2) → 14×14
 ├───┼───┼───┼───┤
 │ ▓ │ ▓ │ ▓ │   │
 └───┴───┴───┴───┘
```

### Loss functions
| Task | Loss | Model output | Target |
|---|---|---|---|
| Regression | `nn.MSELoss` / `nn.L1Loss` | `(B, 1)` or `(B,)` float | same shape, float |
| Multi-class classification | `nn.CrossEntropyLoss` | **raw logits** `(B, C)` | class indices `(B,)` **int64** |
| Binary / multi-label | `nn.BCEWithLogitsLoss` | logits `(B,)` / `(B, C)` | float 0/1, same shape |

`CrossEntropyLoss = LogSoftmax + NLLLoss`. **Do not put a softmax at the end of the model** when using it (it would be applied twice and gradients would get small). Sanity check: an untrained 10-class model has loss ≈ ln(10) ≈ 2.30.

### The training loop (write this on the board, students should memorise it)
```python
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        ...  # validation loss / accuracy
```

### Discussion questions
- Why shuffle the training data but not the validation data?
- Why does `CrossEntropyLoss` want `int64` class indices rather than one-hot vectors?
- A model outputs probabilities via softmax and you feed them into `CrossEntropyLoss`. What goes wrong?

---

## B6 · Live coding: FashionMNIST end-to-end (2:05 – 2:45) — NB02

Run NB02 cell by cell. Students type along or run the provided notebook.

**Suggested flow and what to emphasise**
1. **Config cell:** all hyper-parameters in one place, seeds, `device`, `QUICK_RUN`.
2. **Mean/std** computed on the train set, then transforms.
3. **Train/val/test split** with a seeded generator (54k / 6k / 10k). Explain why the test set is touched once.
4. **Look at the data** before training.
5. **Models:** MLP (≈235k params) vs SmallCNN (≈421k params). Do the **shape walk-through** cell. It is the #1 debugging tool for `mat1 and mat2 shapes cannot be multiplied`.
6. **`train_one_epoch` / `evaluate` / `fit`:** point out `model.train()` vs `model.eval()` and `@torch.no_grad()`, plus keeping the best weights with `copy.deepcopy(model.state_dict())`.
7. Train the MLP (3 epochs), then the CNN (5 epochs). Talk about learning curves while the CNN trains.
8. **Save → reload → verify identical outputs → test accuracy.**
9. **Confusion matrix + misclassified images:** Shirt vs T-shirt/Coat/Pullover are the hard classes.
10. **Comparison table:** params, accuracy, train time, file size (≈ params × 4 bytes). This is the bridge to Day 2 quantization.

**Reference results** (laptop CPU, full run; your exact numbers will vary slightly by hardware):
| Model | Params | Test acc. | Train time |
|---|---|---|---|
| MLP (3 epochs) | 235,146 | ≈ 87% (measured 86.8%) | ≈ 35 s (≈ 12 s/epoch) |
| SmallCNN (5 epochs) | 421,642 | ≈ 91% (measured 91.2%) | ≈ 1.5–2.5 min (≈ 20–28 s/epoch) |

Measured on the reference laptop CPU (PyTorch 2.9.1): whole notebook ≈ 3–3.5 min. The lab notebook (all solutions) takes ≈ 5 min when run top to bottom. Because of the fixed seeds, re-runs on the same machine reproduce these numbers exactly. Per-class test accuracy: Shirt ≈ 70% is the weakest class, followed by T-shirt/top ≈ 84% and Coat ≈ 86%. Bag, Sandal and Trouser are ≥ 97%.

### Discussion questions
- The CNN has more parameters than the MLP here. Is it better *because* it is bigger? (Mostly no. Try a wider MLP: it will not reach CNN accuracy. The inductive bias matters.)
- Why is the CNN slower per epoch than its parameter count suggests? (FLOPs ≠ parameters: each conv weight is reused at every pixel position.)
- Early in training, validation accuracy is sometimes *higher* than training accuracy. Why? (Dropout is active during training. Train metrics are averaged over an epoch while weights change.)

---

## B7 · Generalisation, checkpoints, devices, reproducibility (2:45 – 3:00)

### Overfitting vs underfitting
```
 loss                                        loss
  │ \                                         │ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾  val
  │  \__        ___----  val  ← overfitting   │ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾  train
  │     \______/                              │                   ← underfitting:
  │   \_____                                  │                     both high & flat
  │         \_________  train                 │
  └──────┬──────────────► epochs              └──────────────────► epochs
     best epoch → early stopping / checkpoint
```
- **Underfitting:** bigger model, train longer, higher LR, fix bugs.
- **Overfitting:** more data / **augmentation**, **dropout**, **weight decay**, **BatchNorm** (a mild regulariser), smaller model, **early stopping**.

### Train / validation / test
```
 60,000 official train images                         10,000 official test images
 ┌──────────────────────────────────────┬──────────┐  ┌──────────────┐
 │ train (54k): gradients               │ val (6k) │  │ test (10k)   │
 │                                      │ tuning,  │  │ final report │
 │                                      │ best     │  │ ONCE         │
 │                                      │ epoch    │  │              │
 └──────────────────────────────────────┴──────────┘  └──────────────┘
```

### `train()` vs `eval()` mode
- Dropout: active in `train()`, identity in `eval()`.
- BatchNorm: batch statistics and running-average updates in `train()`, stored running averages in `eval()`.
- `eval()` does **not** disable gradients. Also use `torch.no_grad()` / `inference_mode()`.

### Saving and loading
```python
torch.save(model.state_dict(), "models/fashion_cnn.pth")      # save weights only (recommended)

model = SmallCNN()                                             # need the class definition
model.load_state_dict(torch.load("models/fashion_cnn.pth", map_location="cpu", weights_only=True))
model.eval()
```
- `state_dict` = ordered dict *parameter/buffer name → tensor*. It is portable and robust to refactoring. Pickling the whole model ties the file to your exact code layout.
- For **resuming training**, also save the optimizer's `state_dict` and the epoch (Lab 2 bonus).
- `weights_only=True` (default since PyTorch 2.6) refuses to unpickle arbitrary objects, which is safer for downloaded files.
- `map_location` loads GPU-trained weights on a CPU machine (important for **Day 3 serving** and **Day 4 containers**).

### GPU / device handling
- One `device` variable. Move the **model once**, move **every batch** inside the loop.
- `.item()`, `.cpu()`, `.numpy()` force a GPU→CPU sync. Avoid them in tight loops except for logging.
- On a GPU, `DataLoader(pin_memory=True, num_workers>0)` speeds up transfer. Mixed precision (`torch.autocast`) is a later topic.

### Reproducibility
```python
random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
DataLoader(..., generator=torch.Generator().manual_seed(s))     # reproducible shuffling
random_split(..., generator=torch.Generator().manual_seed(s))   # reproducible split
# GPU bit-exactness additionally: torch.backends.cudnn.deterministic = True; benchmark = False
```
Seeds give repeatability **on the same hardware and library versions**, not across machines. Report results as mean ± std over several seeds.

### Discussion questions
- You tuned 50 hyper-parameter configurations on the validation set. Is your validation accuracy still an unbiased estimate? What is the test set for?
- Why save the *best* epoch rather than the *last* one?
- A colleague's `.pth` file fails with `Missing key(s) in state_dict`. What are two likely causes? (Different architecture/class code, or a `module.` prefix from `DataParallel`.)

---

## ☕ Break 2 (3:00 – 3:10)

---

## B9 · 🧪 STUDENT LAB 2 (3:10 – 4:30) — NB03

Students work in pairs. Every exercise uses a 10k-image subset so each run takes seconds to about a minute on a CPU.

| # | Exercise | Est. time | Concept |
|---|---|---|---|
| 1 | Compare SGD / SGD+momentum / Adam 1e-3 / Adam 0.1 | 12 min | optimizers, LR |
| 2 | `ImprovedCNN` with BatchNorm + Dropout2d; eval vs train mode | 12 min | regularisation, model modes |
| 3 | Implement an `EarlyStopping` class with best-weight restore | 15 min | overfitting, checkpointing |
| 4 | Data augmentation (flip, affine), visualise, train | 12 min | augmentation, train-only transforms |
| 5 | Write a custom `MoonsDataset`, train an MLP, plot the decision boundary | 15 min | `Dataset` API |
| 6 | StepLR vs CosineAnnealingLR vs constant LR | 10 min | LR schedulers |
| ⭐ | Save a full checkpoint (model + optimizer + epoch) and resume | 10 min | checkpoint/resume |

**Trainer tips**
- TODO cells are written to **run without errors** before they are filled in (they print "⏳ not implemented yet"), so "Run All" always works.
- Solutions are directly under each exercise, under **"✅ Solution (reveal after trying)"**. In JupyterLab/VS Code the solution cell's source is collapsed via metadata. Ask students to try for at least 5 minutes first.
- Expected observations:
  - Ex 1: Adam 1e-3 and SGD+momentum learn quickly, plain SGD 0.01 is slow, and **Adam 0.1 fails** (loss ≈ 2.30 = ln 10, ~10% accuracy: it predicts a single class).
  - Ex 2: BN usually speeds up early training. Evaluating in `train()` mode gives different (usually worse) numbers.
  - Ex 3: with only 2k training images (Adam 3e-3), val loss bottoms out around epoch 8 while train loss keeps falling. Early stopping (patience 3) triggers around epoch 11 and restores the best weights.
  - Ex 4: after only 3 epochs, augmentation gives clearly **lower** accuracy (reference run: ≈78% vs ≈85% val). The task became harder and needs more epochs, but train accuracy drops even more, meaning much less overfitting. Use this to discuss why regularisers can look bad in short runs, and why vertical flips are a bad idea for clothing.
  - Ex 6: schedulers help the final epochs settle. The LR plot makes the schedule visible.

---

## B10 · Lab debrief (4:30 – 4:45)
- Ask 2–3 pairs to show one plot each and explain it.
- Walk through the **EarlyStopping** solution (the most error-prone: remember `deepcopy`!).
- Summarise with the class which techniques are about **optimisation** (optimizer, LR, schedulers, BN) versus **generalisation** (dropout, augmentation, early stopping, weight decay).

---

## B11 · Recap, homework & Day 2 teaser (4:45 – 5:00)

### Recap: the 10 things to remember
1. Tensor = data + **shape + dtype + device**.
2. Broadcasting is powerful. Check shapes to avoid silent bugs.
3. Autograd builds a **dynamic graph**. `backward()` fills `.grad` via the chain rule.
4. **zero_grad → forward → loss → backward → step.**
5. `nn.Module` owns parameters. Call `model(x)`.
6. `CrossEntropyLoss` takes **logits** and **int64** targets.
7. `Dataset` (what) + `DataLoader` (how: batch, shuffle).
8. **Train / val / test.** Choose on val, report on test once.
9. `model.train()` / `model.eval()` + `torch.no_grad()`.
10. Save/load **`state_dict`**. Seed everything.

### Homework (≈ 1–2 h)
1. **Finish** any Lab 2 exercises you skipped.
2. **Beat the baseline:** combine BatchNorm, augmentation, a scheduler and early stopping to train on the *full* FashionMNIST training set. Target ≥ 92% test accuracy. Report your configuration, curves and test accuracy in a short markdown cell.
3. **Reading:** PyTorch tutorial *"Learn the Basics"* (pytorch.org/tutorials), and the `torchvision.models` documentation page for **MobileNetV2** (preparation for Day 2).
4. **Think:** our CNN file is ≈ 1.7 MB for 421k parameters. How big would a 3.5M-parameter MobileNetV2 be in float32? And in int8?

### 🔜 Teaser: Day 2 — Optimize
> *"Today we trained a CNN from scratch and hit ~91% on 28×28 greyscale images in a few minutes. Tomorrow's data is CIFAR-10: 32×32 **colour** images of planes, cars, birds and cats, which is much harder. Instead of starting from random weights, we'll take **MobileNetV2 pretrained on ImageNet** (1.2M images), replace its last layer and **fine-tune** it: that's **transfer learning**. Then we make it production-ready: **quantization** turns float32 weights into int8, so the model is about 4× smaller and faster on CPU. We'll measure the accuracy we lose (spoiler: very little)."*

Everything from today carries over: `Dataset`/`DataLoader` (now with ImageNet normalisation and resizing), the same `train_one_epoch`/`evaluate`, `state_dict` saving (with `map_location` for CPU), and the comparison table (size, latency, accuracy), which becomes the core of the optimisation report.

---

## 🧰 Troubleshooting cheat-sheet

| Error message (abridged) | Likely cause | Fix |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (64x3136 and 1568x128)` | wrong `in_features` after flatten | print shapes layer by layer |
| `Expected all tensors to be on the same device` | model or batch not moved | `x = x.to(device)`; `model.to(device)` |
| `expected scalar type Long but found Float` | float labels to `CrossEntropyLoss` | `y.long()` |
| `Found dtype Double but expected Float` | NumPy float64 input | `.float()` |
| `element 0 of tensors does not require grad` | computed loss under `no_grad`, or detached | remove `no_grad`/`detach` from the training path |
| `grad can be implicitly created only for scalar outputs` | `backward()` on a vector | `.mean()` / `.sum()` first |
| `Can't call numpy() on Tensor that requires grad` | converting a tracked tensor | `.detach().cpu().numpy()` |
| DataLoader hangs / `BrokenPipeError` on Windows | `num_workers>0` in Jupyter | `num_workers=0` |
| Loss stuck at ≈ 2.30 (10 classes) | LR too high, dead network, or labels shuffled | lower LR, check data/labels |
| Loss `nan` | LR too high, `log(0)`, bad normalisation | lower LR, check inputs |
| `Missing key(s) in state_dict` | architecture mismatch / `module.` prefix | use the same class; strip prefix |
