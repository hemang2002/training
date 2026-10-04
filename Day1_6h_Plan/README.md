# Day 1 — 6-hour plan: Check → Show → Solve → Conclude

> **Length:** 6 h (09:30 – 15:30) = **4 h 30 min teaching + 1 h 30 min breaks**
> **Audience:** MSc students. Some have never trained a network; others have.
> **Goal of the day:** every student trains a CNN on FashionMNIST and can explain each line of the training loop.

This folder **replaces the timetable** in [`Day1_PyTorch_Fundamentals/README.md`](../Day1_PyTorch_Fundamentals/README.md).
The concept notes, diagrams and discussion questions in that README are still valid; use them as background.

## 📁 What is in this folder

| File | What it is | Used in |
|---|---|---|
| [`00_check_in.html`](00_check_in.html) | Quiz page, works offline. **Morning check-in** (15 questions, 5 topics) and **end-of-day exit quiz** (10 questions). Students get a **score code** like `3-2-1-0-1` | B0, B5 |
| [`01_beat_the_cnn.html`](01_beat_the_cnn.html) | 🎁 **Surprise game: "Can you beat the CNN?"** Students label real FashionMNIST test pictures, then compare with the real answers of the Day-1 CNN. Data: `game_data.js`, made by `make_game_data.py` | B4 |
| [`day1_solve_tasks.ipynb`](day1_solve_tasks.ipynb) | One "Solve" task per block (Tasks 1–5), each with hints, TODO cells and hidden solutions. Task 5 is the final challenge | B1–B5 |
| `README.md` | This plan | — |

It also uses existing material:
- the notebooks in `Day1_PyTorch_Fundamentals/`
- the HTML lessons in `Training_Visualized/` (open `index.html`)

---

## 🔁 Every block follows the same 4 steps

| Step | Time | What happens | Why |
|---|---|---|---|
| 🔍 **Check** | 3–5 min | 2–3 quick questions, answered with hands up (answers are below each block) | You learn what they already know, and their brains "open the drawer" for the topic |
| 👀 **Show** | 15–30 min | First an HTML lesson (**feel it**), then the notebook (**see it in real code**) | Example first, then the name, then the code |
| ✍️ **Solve** | 15–25 min | A task from `day1_solve_tasks.ipynb`, in **pairs** | They prove to themselves that they can do it |
| ✅ **Conclude** | 3–5 min | One sentence on the board + one "exit question" | Students leave each block with **one** thing they remember |

**Bridge sentences.** Whenever you switch from HTML to notebook, say out loud how they connect, e.g. *"Remember the ball rolling downhill? This line is that step."*
Use the HTML and the notebook one after the other, not side by side.

---

## 🗓️ Timetable

| Time | Block | Check | Show | Solve | Min |
|---|---|---|---|---|---|
| 09:30 – 09:45 | **B0** Welcome + 🌅 check-in quiz | `00_check_in.html` | course roadmap | — | 15 |
| 09:45 – 10:30 | **B1** 🧱 Tensors | 3 questions | NB01 Part 1 | **Task 1** | 45 |
| 10:30 – 10:45 | ☕ **Break** | | | | 15 |
| 10:45 – 11:40 | **B2** ⛰️ How a model learns | 3 questions | HTML 01 → 02 → 03, then NB01 Parts 2–4 | **Task 2** | 55 |
| 11:40 – 12:40 | 🍽️ **Lunch** | | | | 60 |
| 12:40 – 13:40 | **B3** 🔁 The training loop + a real CNN | 3 questions | HTML 04 → 10, then NB02 §1–5 | **Task 3** | 60 |
| 13:40 – 14:40 | **B4** 🎯 Is the model any good? | 🎁 **surprise game** | NB02 §6–8, then HTML 07 → 09 | **Task 4** | 60 |
| 14:40 – 14:55 | ☕ **Break** | | | | 15 |
| 14:55 – 15:30 | **B5** 🏁 Final challenge + 🌇 exit quiz + recap | — | — | **Task 5** | 35 |

**Total:** 270 min teaching (B0–B5) + 90 min breaks = **6 h**.
About **135 min** is students working (check-in, surprise game, Tasks 1–5, exit quiz). Most of the rest is interactive HTML that they click themselves.

---

## B0 · Welcome + check-in (09:30 – 09:45)

| Min | Do |
|---|---|
| 3 | Welcome. Show the course map: *train → optimize → serve → containerize → deploy*. Today = **train**. |
| 9 | Students open **`00_check_in.html`** (🌅 Morning tab) and answer all 15 questions **alone**. Each answer is explained right away, so the quiz already teaches. |
| 3 | **Read the room** (see below). |

### 📊 How to read the check-in in 2 minutes

Each student gets a **score code**, one digit per topic: Python & NumPy, Tensors, Learning, Loop, Good model.
Example: `3-2-1-0-1` means strong in NumPy, weak in the training loop.

Ask for hands, topic by topic: *"Who got 3 out of 3 on Tensors? 2? 1 or 0?"* Write the counts on the board.

| What you see | What to do in that block |
|---|---|
| Most students **3/3** | Shorten **Show** by about 5 min. Go straight to Solve and give the ⭐ bonus parts |
| Mixed | Normal plan. **Pair** a strong student with a weaker one (use the "have you trained a network before?" answer) |
| Most students **0–1/3** | Use HTML **Simple view**, go slower in Show, and do Solve **together** on the projector for the first half |

Keep the board counts. In B5 the exit quiz uses the same topics, so the class can see what it gained.

---

## B1 · 🧱 Tensors (09:45 – 10:30)

**🔍 Check (5 min)**, hands up:
1. *"What are the three things you must always know about a tensor?"* → **shape, dtype, device**
2. *"(4×3) @ (3×5) gives which shape?"* → **(4, 5)**
3. *"Batch of 32 RGB images, 28×28 — which shape in PyTorch?"* → **(32, 3, 28, 28)**

**👀 Show (17 min)**: `Day1_PyTorch_Fundamentals/01_tensors_and_autograd.ipynb`, **Part 1**. Type along together.
- §1.2 shape / dtype / device: *"a tensor is a NumPy array that knows where it lives and can remember its history"*
- §1.3 reshaping (`view`, `permute`)
- §1.7 **broadcasting**: draw the "stretch the size-1 axis" picture on the board
- §1.8 matmul: the "inner sizes must match" rule
- §1.9 devices (1 min: `.to(device)`)

**✍️ Solve (18 min)**: **Task 1** in `day1_solve_tasks.ipynb`
- 1a *Predict the shape*: write your guess first, then run the checker.
- 1b *Fix the broadcasting bug*: the code runs without an error, but the answer is wrong. That is the dangerous kind of bug.

**✅ Conclude (5 min)**
- Board: **"Check the shape. Broadcasting never complains, it just does something."**
- Exit question: *"Why is a silent bug worse than an error?"*

---

## B2 · ⛰️ How a model learns (10:45 – 11:40)

**🔍 Check (5 min)**:
1. *"What does the loss tell you?"* → **one number for how wrong the model is**
2. *"You're on a foggy hill and want the valley. Which way do you step?"* → **downhill, against the gradient**
3. *"What goes wrong if the steps are too big?"* → **you jump over the valley; the loss bounces or explodes**

**👀 Show (25 min)**: HTML first, then code.

| Min | Material | What students do |
|---|---|---|
| 7 | `Training_Visualized/01_what_is_a_model.html`: 🍦 ice-cream story + "fit the line by hand" | Turn the knobs and watch the loss number |
| 8 | `02_gradient_descent.html`: Part A, one knob, one ball | Move the **learning-rate** slider: too small, just right, too big |
| 4 | `03_backpropagation.html`: 🎬 step through | See where the gradient comes from |
| 6 | NB01 **Parts 2–4**: `requires_grad`, `backward()`, `.grad`, gradients accumulate, `no_grad()`, then gradient descent from scratch → `nn.Linear` + `optim` | **Bridge:** *"the ball in page 02 = this update line"* |

**✍️ Solve (20 min)**: **Task 2**
- 2a Autograd vs your own derivative
- 2b **Learning-rate hunt**: they discover the "bounces forever" and "explodes" cases themselves
- 2c Fit a line with a manual loop

**✅ Conclude (5 min)**
- Board: **"Loss = how wrong. Gradient = which way is uphill. Step downhill, not too big."**
- Exit question: *"With lr = 1.0 on (w − 3)², why does w bounce between 0 and 6?"*

---

## B3 · 🔁 The training loop + a real CNN (12:40 – 13:40)

**🔍 Check (5 min)**:
1. *"What is an epoch? A batch?"* → **one full pass / a small group used for one update**
2. *"Why `zero_grad()`?"* → **PyTorch adds new gradients on top of old ones**
3. *"Say the 5 steps of the loop."* (Most can't yet, and that's the point.)

**👀 Show (30 min)**

| Min | Material | Focus |
|---|---|---|
| 8 | `04_training_loop.html` | Run it, then flip the **bug switches** (forget `zero_grad`, wrong lr…) and watch training break |
| 7 | `10_cnn_convolution.html`: draw a digit, pick a kernel | *"A CNN slides small patterns over the picture"* |
| 15 | `Day1_PyTorch_Fundamentals/02_train_cnn_fashionmnist.ipynb` **§1–5**: data, DataLoader, `nn.Module`, loss/train/eval functions, train the MLP and the CNN | Start the CNN training, then explain the loop **while it runs**. If time is short, set `QUICK_RUN = True` |

Write the loop on the board. Students should be able to say it by heart:

```
for xb, yb in loader:
    optimizer.zero_grad()      # 1. clean the slate
    out  = model(xb)           # 2. guess
    loss = loss_fn(out, yb)    # 3. how wrong?
    loss.backward()            # 4. who is to blame?
    optimizer.step()           # 5. fix the knobs
```

**✍️ Solve (20 min)**: **Task 3**
- 3a Put the scrambled loop in order
- 3b *What if we forget `zero_grad`?* Train twice and compare the curves

**✅ Conclude (5 min)**
- Board: **"zero_grad → forward → loss → backward → step."** The whole class says it out loud once.
- Exit question: *"Which line do we skip in evaluation, and why?"* → **backward and step: we only measure, we don't learn**

---

## B4 · 🎯 Is the model any good? (13:40 – 14:40)

### 🎁 Surprise first: "Can you beat the CNN?" (25 min, replaces the Check)

**Don't announce it in the morning.** Say only: *"After lunch, you'll compete against something."*

| Min | Do |
|---|---|
| 2 | Everyone opens **`01_beat_the_cnn.html`**: *"This morning you built a CNN. Now let's see if you're better than it."* |
| 5 | **Round 1**: 20 test pictures, all 10 kinds, alone and in silence. The page also times them. |
| 5 | Results screen. **Hands up:** *"Who beat the CNN? Draw? Lost?"* Click **"❌ Both wrong"**: *"Why did both of you fail on these?"* |
| 4 | **Round 2**: 15 pictures, only T-shirt / Pullover / Coat / Shirt. |
| 9 | Results + **the two confusion matrices** (yours vs the CNN's). Discussion: |

Discussion points:
- *Where are your red cells?* Almost always in the **Shirt** row, and the CNN's too.
- *Is the model bad, or the data hard?* At 28 × 28 pixels a shirt and a pullover can look the same. When humans fail on the same pictures, the **data** is the limit.
- *Speed:* they needed a minute or two for 35 pictures. The CNN labels **all 10,000 test pictures in about 1.4 s** (measured on a laptop CPU; a busy or slower laptop may need a few seconds, which is still far faster than any human).
- **Bridge:** *"You just made a confusion matrix by hand. Now let's read the CNN's on all 10,000 pictures."* → Show.

**What the CNN scores on the game's pictures:** 17/20 in round 1 and 12/15 in round 2. These are the real predictions of `Day1_PyTorch_Fundamentals/models/fashion_cnn.pth`, which gets 91.2% on the full test set and 82% on the four top classes. Many students beat it in round 1; round 2 is close.

**👀 Show (15 min)**

| Min | Material | Focus |
|---|---|---|
| 7 | NB02 **§6** learning curves + **§8** confusion matrix and misclassified pictures | *"Same Shirt row as in your game, now with 10,000 pictures"* |
| 4 | `07_overfitting.html`: ⏱️ early stopping | Find the moment to stop |
| 4 | `09_metrics.html`: ① the accuracy trap | Slide "how many are sick" down to 1% |

**✍️ Solve (15 min)**: **Task 4**
- 4a Early stopping by hand: best epoch, and when to stop with patience 2
- 4b Recall per class from the **real** confusion matrix: find the weakest class

**✅ Conclude (5 min)**
- Board: **"Choose on validation, report on test once. One number can hide a weak class."**
- Exit question: *"What would you tell a manager who only asks for accuracy?"*

---

## B5 · 🏁 Final challenge + exit quiz (14:55 – 15:30)

| Min | Do |
|---|---|
| 22 | **Task 5: "Five bugs"**. One training script with five bugs, one from each block of the day. Pairs fix the bugs until the test accuracy reaches **at least 93%** (the fixed version measured **95.0%**). Give the hints only after 10 minutes. |

**What students will see in Task 5** (measured):
- **As given:** the script crashes at once with `expected scalar type Long but found Float`. Labels must be int64 (bug 3: the dtype, from B1).
- **Only that bug fixed:** it runs, but test accuracy is **44.7%**. Four bugs are left:
  - no `zero_grad`
  - softmax before `CrossEntropyLoss`
  - no `eval()` / `no_grad()`
  - learning rate 1.0
- **Softmax bug:** on its own it doesn't lower the accuracy here; its symptom is a train loss **stuck at about 0.36**. This is a good discussion point: *"Accuracy looks fine, so is the bug harmless?"*
| 5 | 🌇 **Exit quiz**: `00_check_in.html`, **End-of-day** tab (10 questions). On the same browser, the page shows *"This morning you scored X%. Now: Y%."* |
| 8 | Hands up per topic again, and compare with this morning's counts on the board. Then the **recap** and **homework** below. |

### 🎓 Recap: 5 things to remember for life
1. **Tensor = data + shape + dtype + device.** Check shapes: broadcasting never complains.
2. **Loss** says how wrong, the **gradient** says which way is uphill, and the **learning rate** says how big a step.
3. **zero_grad → forward → loss → backward → step.**
4. **`model.eval()` + `torch.no_grad()`** before measuring.
5. **Choose on validation, report on test once.** Look beyond accuracy: the confusion matrix and the per-class recall.

### 🏠 Homework (≈ 1–1.5 h)
- `Day1_PyTorch_Fundamentals/03_student_lab.ipynb` (the old Lab 2):
  - Exercises **1** (optimizers), **2** (BatchNorm + Dropout) and **3** (EarlyStopping class)
  - the others are optional
- Click through `Training_Visualized/05_activation_functions.html`, `06_neural_network_playground.html` and `08_dropout_batchnorm.html`.

---

## ⏱️ If you run late / early

| Situation | Do this |
|---|---|
| **Late by 10 min** | In B2, skip `03_backpropagation.html`. In B3, set `QUICK_RUN = True` |
| **Late by 20+ min** | Shorten the game to round 2 only (10 min). Task 2c and Task 4a become homework. Task 5: do bugs 1–3 together on the projector |
| **Early** | ⭐ Bonus parts in the tasks; NB02 CNN for 10 epochs; `06_neural_network_playground.html` |
| **A strong group** (check-in mostly 3/3) | Start Task 5 earlier and add Lab 2 Exercise 2 (BatchNorm + Dropout) from `03_student_lab.ipynb` |

## ✅ Before the session
- [ ] Run `day1_solve_tasks.ipynb` and NB02 once on the presentation machine. This downloads FashionMNIST, so it isn't downloaded live.
- [ ] Select the course kernel (`.venv` / "Python (ml-course)"), not the default Anaconda one.
- [ ] Open `00_check_in.html` and `Training_Visualized/index.html` in the browser tabs you will use.
- [ ] Students: environment installed and the first cell of NB01 runs (see the root `README.md`).

## What changed compared with the old 5-hour plan

| Old plan | New plan |
|---|---|
| 5 h, 20 min of breaks | 6 h, **90 min** of breaks (two 15-min breaks + lunch) |
| Lecture first, labs in two big blocks | Every block: **Check → Show → Solve → Conclude** |
| No surprise element | 🎁 **"Beat the CNN"** game: humans vs their own model, with real predictions |
| No idea of the starting level | **Check-in quiz** at 09:30, **exit quiz** at 15:20, with a before/after comparison |
| Lab 2 (80 min, 6 exercises) in class | Short **Tasks 1–5** in class; Lab 2 Exercises 1–3 become **homework** |
| Theory explained with words | HTML lessons first (students click), then the notebook |
