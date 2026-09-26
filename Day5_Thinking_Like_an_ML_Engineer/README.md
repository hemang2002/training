# Day 5 — Thinking Like an ML Engineer

> **Trainer guide.** Day 5 has no new tooling. It is a structured, question-driven day that turns four days of "how" into "why", "what if" and "what would you do". Students should leave able to reason about trade-offs, debug systematically, estimate costs quickly, and defend design decisions.

---

## 1. Learning objectives

By the end of the day, students should be able to:

| # | Objective | Evidence during the day |
|---|-----------|-------------------------|
| 1 | Explain *why* core techniques work (autograd, BatchNorm, transfer learning, quantization, pruning) rather than recite *how* | "Why?" deep-dive answers survive two follow-ups |
| 2 | Debug ML systems methodically: form hypotheses, pick the cheapest test, isolate layers (data → model → export → serving → infra) | Debugging scenario write-ups |
| 3 | Make back-of-envelope estimates for memory, latency, throughput, bandwidth and cost | Fermi round within ±3× of reference |
| 4 | Design an end-to-end ML system under real constraints and justify trade-offs | System-design presentations |
| 5 | Recognise production risks: drift, calibration, bias, privacy, cost blow-ups | Debate round and quiz |
| 6 | Connect all 5 days into a single mental model: **data → train → optimize → serve → package → orchestrate → deploy → monitor** | Wrap-up "one-page map" |

---

## 2. Files in this folder

| File | Use it for | Contents |
|------|-----------|----------|
| [`question_bank.md`](question_bank.md) | Rapid-fire, "why?" deep dives, cold calls | 106 questions across 10 topics, each with difficulty, model answer, follow-ups |
| [`debugging_scenarios.md`](debugging_scenarios.md) | Group debugging block | 15 broken scenarios with symptoms, progressive hints, root cause, fix, lesson |
| [`system_design_cases.md`](system_design_cases.md) | Design mini-case block | 5 case studies with requirements, guiding questions, reference solution sketches |
| [`fermi_estimation.md`](fermi_estimation.md) | "Estimate it" round | 10 estimation problems with worked solutions |
| [`final_quiz.md`](final_quiz.md) | Closing quiz | 30 questions (MCQ, T/F, short answer) + answer key |

---

## 3. Timetable (5 hours, relative times)

Legend: **[CORE]** = do not skip. **[FLEX]** = if time permits; shrink or drop first.

| Time | Block | Format | Materials | Tag |
|------|-------|--------|-----------|-----|
| 0:00 – 0:10 | **Kick-off**: agenda, team formation (groups of 3–4), scoring rules | Plenary | This README §6 | CORE |
| 0:10 – 0:30 | **Warm-up rapid-fire** — 15–20 ★ questions, 30–45 s each, teams buzz in | Whole class, gamified | `question_bank.md` (★ items) | CORE |
| 0:30 – 1:10 | **"Why?" deep dives** — 6–8 ★★/★★★ questions, each pushed through 2–3 Socratic follow-ups | Cold call + think-pair-share | `question_bank.md` (topics 1–4) | CORE |
| 1:10 – 1:20 | **Break 1** | — | — | — |
| 1:20 – 2:20 | **Debugging scenarios** — each group gets 2 scenarios (≈20 min each), then 2-min report-back per scenario | Small groups (3–4) | `debugging_scenarios.md` | CORE |
| 2:20 – 2:50 | **"Estimate it" Fermi round** — 4–6 problems, 4 min each, teams write answer on paper, closest wins | Teams | `fermi_estimation.md` | CORE (4 problems) / FLEX (extra) |
| 2:50 – 3:05 | **Break 2** | — | — | — |
| 3:05 – 3:55 | **System-design mini-case** — 25 min design, 4 min presentation + 2 min critique per group | Groups (4–5) | `system_design_cases.md` | CORE |
| 3:55 – 4:15 | **Debate / opinion round** — 2–3 motions, 6 min each | Two sides + judges | §5 below | FLEX |
| 4:15 – 4:40 | **Final quiz** — 30 questions, individual, then 5-min answer walk-through of the hardest 5 | Individual | `final_quiz.md` | CORE |
| 4:40 – 5:00 | **Wrap-up & feedback** — "one-page map" of the course, leaderboard, feedback form, next steps | Plenary | §7 below | CORE |

### Stretch / shrink rules

| If you are… | Do this |
|-------------|---------|
| **Running 30 min late** | Drop the debate round (–20 min), cut Fermi to 3 problems (–10 min) |
| **Running 60 min late** | Also cut debugging to 1 scenario per group (–20 min) and quiz walk-through (–5 min); run system design with 3 presenting groups only |
| **Ahead of schedule** | Add ★★★ questions from topics 9–10 (MLOps, ethics); run a second design case in "speed mode" (10 min design, whiteboard only); add "10 reasons" brainstorm (Q9.1) |
| **Energy is low** (post-lunch) | Switch to a physical format: stand-up rapid-fire, "walk to the wall that matches your answer" true/false |
| **One group dominates** | Use "only people who haven't answered yet" rule, or rotate the group's spokesperson |

---

## 4. Facilitation tips

### 4.1 How to ask

- **Ask, then name.** "Why does INT8 quantization only lose ~1% accuracy? … Priya." Everyone thinks before knowing who is called on.
- **Prefer open "why/what if/how would you know" over "what is".** "What is a readiness probe?" is trivia. "What goes wrong if readiness and liveness hit the same endpoint?" is engineering.
- **Anchor to their own code from Days 1–4.** "In your Day 3 API, where exactly would you put this check?" makes abstract ideas concrete.
- **One question at a time.** Do not stack sub-questions; keep follow-ups in your pocket.

### 4.2 Wait time

- Wait **at least 5 seconds** after a question before rephrasing or answering. Silence is thinking, not failure.
- For ★★★ questions, give **30–60 seconds of silent individual thinking** or **think-pair-share** (1 min alone, 2 min with a neighbour, then share).
- After a student answers, wait **3 more seconds** — they often add the best part themselves.

### 4.3 Socratic follow-ups (use on any answer)

| Move | Example prompt |
|------|----------------|
| Clarify | "What do you mean by 'the model is slow' — latency, throughput or startup?" |
| Probe assumptions | "You assumed the GPU is always busy. Is it at 3 a.m.?" |
| Evidence | "How would you *measure* that? What would the graph look like?" |
| Counter-example | "Then why is our quantized model *slower* on this ARM laptop?" |
| Consequence | "If that's true, what else must be true?" |
| Change a constraint | "Now the budget is zero / RAM is 512 MB / there is no internet." |
| Meta | "What question should we have asked first?" |

### 4.4 Handling wrong answers

1. **Thank and extract the useful part**: "Good — you're right that fewer bits means less precision. Let's test the second half of that claim."
2. **Turn it into a test**, not a verdict: "If that were the reason, what would happen when we…?" Let the student discover the contradiction.
3. **Open it to the room**: "Who has a different hypothesis?" — never "Who can give the *right* answer?"
4. **Normalise**: tell a story about a real mistake you made (e.g., shipping a model with the wrong normalization).
5. **Return to the student later** with an easier question they will get right.

### 4.5 Group sizes

| Activity | Group size | Roles |
|----------|-----------|-------|
| Rapid-fire | Teams of 3–4 | One "buzzer" per round, rotating |
| Debugging | 3–4 | Driver (reads hints), Skeptic (challenges hypotheses), Scribe, Reporter |
| System design | 4–5 | Architect, Cost analyst, Risk/ethics owner, Presenter, (Timekeeper) |
| Debate | 2 sides × 2–3 + remaining as judges | Opening, rebuttal, closing |

### 4.6 Scoring & gamification

| Activity | Points |
|----------|--------|
| Rapid-fire correct | +1 (★), steal on wrong answer +1 |
| Deep-dive: answer survives 2 follow-ups | +2; best follow-up *question* from the audience +1 |
| Debugging: root cause found with 0 / 1 / 2 / 3 hints | +5 / +4 / +3 / +2; correct fix +1; lesson articulated +1 |
| Fermi: within 2× of reference | +3; within 3× +2; within 10× +1; best-explained reasoning +2 (reasoning > number) |
| System design (peer-voted + trainer) | up to 10: requirements coverage 2, architecture 2, trade-offs 2, monitoring 2, presentation 2 |
| Debate (judges vote) | +3 winning side, +1 best single argument |
| Final quiz | individual score; team bonus = team average / 3 |

Ideas: a live leaderboard on the whiteboard; "golden question" cards (double points) for ★★★; a "hint token" each team can spend once; small prizes (stickers, choosing next break length). Reward **reasoning quality** explicitly — say "+1 for how you got there" out loud.

---

## 5. Debate / opinion motions (pick 2–3)

Each side gets 90 s opening, 60 s rebuttal; judges then vote. Push both sides to quantify.

1. *"This house believes every production model should be quantized to INT8 by default."*
2. *"Kubernetes is overkill for 95% of ML student and startup projects."*
3. *"Accuracy on a held-out test set is a misleading metric for deployment decisions."*
4. *"Managed platforms (HF Spaces, Render) beat self-hosting (VM + k3s) for small teams."*
5. *"GAN-generated faces should always be watermarked, by law."*
6. *"Model monitoring matters more than model architecture."*
7. *"You should always start with the smallest model that could possibly work."*
8. *"Edge inference is more ethical than cloud inference."*

Trainer notes: there is no right side. Look for arguments that name **constraints** (latency, cost, team size, regulation), **evidence** (benchmarks, incident stories), and **conditions** ("true when X, false when Y"). The best closing statement usually says "it depends on…" and then names exactly what.

---

## 6. Kick-off script (≈3 min)

> "For four days you learned *how* to train, optimize, serve, containerize and deploy. Today you'll learn how to *think* about those things when nobody gives you the recipe. In a real job, nobody asks 'what is quantization?'. They ask 'why is our bill tripled?', 'why do users say the model is wrong?', and 'can this run on a 2 GB box with no internet?'. Every answer today is judged on reasoning, not recall. Wrong answers with good reasoning score points."

---

## 7. Wrap-up & feedback (20 min)

1. **One-page map (8 min).** Draw on the board and have the class fill it in:

```
 Data ──► Train ──► Evaluate ──► Optimize ──► Export ──► Serve ──► Package ──► Orchestrate ──► Deploy ──► Monitor
 (D1)     (D1,D2)   (D1,D2)      (D2)         (D2 ONNX)  (D3 API)  (D4 Docker) (D4 K8s)        (D4 cloud) (D5)
   ▲                                                                                                          │
   └──────────────────────────── drift / feedback / retraining ◄──────────────────────────────────────────────┘
```

   For each arrow ask: *"What is the most common failure at this boundary?"* (e.g. Evaluate→Optimize: accuracy measured on fp32 only; Export→Serve: preprocessing mismatch; Package→Orchestrate: memory limits.)

2. **Leaderboard & prizes (3 min).**
3. **Feedback (5 min).** Collect: one thing to keep, one thing to change, one topic you want to go deeper on, confidence 1–5 before vs after the course for each day's topic.
4. **Next steps (4 min).** Suggest: put the Day 3–4 project on GitHub with a README, a Dockerfile and a live link; add a `/metrics` endpoint and a drift check; read "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., NeurIPS 2015) and "On Calibration of Modern Neural Networks" (Guo et al., ICML 2017).

---

## 8. Trainer preparation checklist

- [ ] Print or share debugging scenario cards (one per group, hints folded/hidden).
- [ ] Prepare a leaderboard (whiteboard or shared sheet).
- [ ] Pick 20 ★ questions for rapid-fire and 8 ★★/★★★ for deep dives in advance; mark them in `question_bank.md`.
- [ ] Assign system-design cases to groups (one per group; duplicates are fine — compare solutions).
- [ ] Have the Day 2–4 repos open to show real code when an answer references it.
- [ ] Set up the quiz (paper, Google Form, or Kahoot for the MCQ/T-F part).
- [ ] Prepare the feedback form link.
