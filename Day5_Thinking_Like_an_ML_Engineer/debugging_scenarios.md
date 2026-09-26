# Day 5 — Debugging Scenarios

**15 realistic "it's broken" scenarios** drawn from the Day 1–4 stack. Each one has symptoms (logs, code, commands), progressive hints, root cause, fix and the lesson.

## How to run this block (60 min)

1. Groups of 3–4. Give each group **2 scenarios** (a card with only the **Symptoms** section).
2. Each group writes down **3 hypotheses ranked by likelihood** and, for each, **the cheapest test** that would confirm or rule it out. *This is the key skill.* Reward it even if the final answer is wrong.
3. A group may request hints one at a time. Each hint costs 1 point (see the README scoring).
4. Report-back (2 min per scenario): symptom → hypotheses → test → root cause → fix → **lesson**.

> **A debugging mantra to put on the board:** *Reproduce → Isolate the layer (data / model / export / serving / container / cluster / cloud) → Hypothesize → Cheapest test → Fix → Add a test so it never returns.*

Hints and answers are inside collapsible blocks (`<details>`), so you can project the file without spoiling them.

## Index

| # | Scenario | Layer | Difficulty |
|---|----------|-------|-----------|
| 1 | Loss becomes NaN after a few hundred steps | Training | ★★ |
| 2 | Transfer learning stuck at 10% accuracy | Training | ★★ |
| 3 | API gives different answers for the same image | Serving / PyTorch | ★★ |
| 4 | The API says everything is a "frog" | Preprocessing | ★★ |
| 5 | "ONNX output differs from PyTorch!" | Export | ★★★ |
| 6 | The INT8 model is slower than FP32 | Optimization | ★★★ |
| 7 | Detector trains "perfectly" but finds no pedestrians | Detection | ★★ |
| 8 | Container runs, but the host can't reach Flask | Docker | ★ |
| 9 | Streamlit can't reach the API in docker-compose | Docker Compose | ★ |
| 10 | `ImagePullBackOff` on minikube/kind | Kubernetes | ★ |
| 11 | `CrashLoopBackOff` with exit code 137 | Kubernetes | ★★ |
| 12 | Pods restart forever during start-up | Kubernetes probes | ★★ |
| 13 | Streamlit in k8s: "Failed to resolve 'api'" | Kubernetes networking | ★★ |
| 14 | HPA shows `<unknown>/70%` and never scales | Kubernetes autoscaling | ★★ |
| 15 | Render demo times out on the first request | Cloud / free tier | ★ |

---

## Scenario 1 — Loss becomes NaN after a few hundred steps

**Context:** Day 1 CNN on FashionMNIST. A student wrote their own loss "to understand cross-entropy".

**Symptoms**
```text
epoch 1 step 100  loss 0.7132
epoch 1 step 200  loss 0.5518
epoch 1 step 300  loss inf
epoch 1 step 400  loss nan
epoch 1 step 500  loss nan      # accuracy drops to 10% and stays there
```
```python
logits = model(x)
probs = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdim=True)
loss = -(F.one_hot(y, 10) * torch.log(probs)).sum(dim=1).mean()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
```

<details><summary>Hint 1</summary>

The loss becomes `inf` *before* it becomes `nan`. Which operation in the loss can produce `inf`?
</details>

<details><summary>Hint 2</summary>

What is `torch.log(0.0)`? When could `probs` be exactly 0 in float32? What happens to `torch.exp(logits)` when a logit is 100?
</details>

<details><summary>Hint 3</summary>

Look at the learning rate. What does a huge step do to the size of the logits?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** a naive softmax and log is numerically unstable. The high LR (0.5) makes the logits grow large. `exp(logit)` overflows to `inf` (float32 overflows above about 88.7), and tiny probabilities underflow to `0`, so `log(0) = -inf`. `inf − inf` or `0 · inf` gives `nan`. One NaN gradient poisons every weight, and the output collapses to a constant (10%).

**Fix**
```python
loss = F.cross_entropy(logits, y)                       # uses log-softmax with the log-sum-exp trick
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # optional safety net
```
To debug: `torch.autograd.set_detect_anomaly(True)` finds the op that produced the NaN. Log the gradient norms.

**Lesson:** use the framework's fused, numerically stable losses (`cross_entropy` takes **logits**, not probabilities). Watch for `inf` before `nan`. Check the LR first. Other NaN sources: dividing by a std of 0 during normalization, `sqrt(0)` in the backward pass, FP16 overflow without a loss scaler, corrupt input data.
</details>

---

## Scenario 2 — Transfer learning stuck at 10% accuracy

**Context:** Day 2 MobileNetV2 on CIFAR-10 at 96 px, frozen backbone.

**Symptoms**
```text
epoch 1  train_loss 2.3071  val_acc 0.0987
epoch 2  train_loss 2.3049  val_acc 0.1012
epoch 3  train_loss 2.3066  val_acc 0.0995
```
```python
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
for p in model.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
model.classifier[1] = nn.Linear(model.last_channel, 10)   # new 10-class head
model.to(device)
# ... standard training loop: zero_grad, forward, loss, backward, step ...
```

<details><summary>Hint 1</summary>

A loss of 2.30 is ln(10). What does it mean when the loss *never moves* from the value it has at random init?
</details>

<details><summary>Hint 2</summary>

Check whether the new head's weights change between epochs: `model.classifier[1].weight.sum()` before and after an epoch.
</details>

<details><summary>Hint 3</summary>

Which parameters does the optimizer hold? Print `[p.shape for g in optimizer.param_groups for p in g['params']]`.
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** the optimizer was built **before** the head was replaced. It holds references to the *old* `Linear(1280→1000)` tensors (which are frozen anyway). The new `Linear(1280→10)` gets gradients, but no optimizer ever updates it. The head stays random, so the loss stays at about ln 10. The small fluctuations come from BN running statistics updating in train mode and from batch noise. PyTorch gives **no warning**.

**Fix:** replace the head first, move the model to the device, then create the optimizer:
```python
model.classifier[1] = nn.Linear(model.last_channel, 10)
model.to(device)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
```

**Lesson:** build the optimizer *after* the model's final structure (and device placement) is set. Sanity checks: count trainable parameters, confirm the ones you expect change after one step, and try to **overfit a single batch** first (loss should go to about 0). A model that can't overfit one batch has a plumbing bug, not a modelling problem.
</details>

---

## Scenario 3 — API gives different answers for the same image

**Context:** an early Day 3 version of the API served the PyTorch model directly.

**Symptoms**
```bash
$ for i in 1 2 3; do curl -s -F file=@cat.jpg localhost:5000/predict; echo; done
{"label": "cat",  "confidence": 0.41}
{"label": "dog",  "confidence": 0.37}
{"label": "frog", "confidence": 0.29}
```
Users also report the model "gets worse through the day".
```python
model = torch.load("mobilenet_cifar.pt", weights_only=False)

@app.post("/predict")
def predict():
    x = preprocess(request.files["file"]).unsqueeze(0)
    with torch.no_grad():
        probs = model(x).softmax(1)
    ...
```

<details><summary>Hint 1</summary>

The model is deterministic maths. Which layers behave *randomly* or *batch-dependently*?
</details>

<details><summary>Hint 2</summary>

`torch.no_grad()` is present. Does it change layer behaviour? (See Q1.3.)
</details>

<details><summary>Hint 3</summary>

What does BatchNorm do to `running_mean` on every forward pass in train mode, even under `no_grad`?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** `model.eval()` is never called. A saved model comes back in whatever mode it was saved in, here train mode.
- **Dropout** (MobileNetV2 classifier, p = 0.2) randomly zeroes activations, so each call gives a different answer.
- **BatchNorm** normalizes with the statistics of the current single image instead of the learned running statistics, which distorts the features.
- It also **updates `running_mean`/`running_var` with every request** (buffers update without gradients). The model slowly drifts toward the recent request distribution. That is why it "gets worse through the day".

**Fix:** `model.eval()` once after loading. Better, export to ONNX from eval mode (Day 3 did this) and run it with ONNX Runtime. A second issue to point out: `torch.load(..., weights_only=False)` unpickles arbitrary objects, so a tampered checkpoint can execute code. Save a `state_dict` instead and load it with `weights_only=True`.

**Lesson:** `eval()` and `no_grad()` do different jobs. Always test determinism: the same input twice must give the same output. Add it to CI.
</details>

---

## Scenario 4 — The API says everything is a "frog"

**Context:** the notebook reaches 94% on CIFAR-10 at 96 px. The ONNX API (Day 3) was written by a different teammate.

**Symptoms**
- 38 of 50 test photos come back as `frog` or `deer`. There are no errors in the logs.
- Running the same ONNX file in the notebook, on the notebook's tensors, gives correct predictions.
```python
# api/preprocess.py
import cv2, numpy as np
def preprocess(path):
    img = cv2.imread(path)                       # HWC, uint8
    img = cv2.resize(img, (96, 96))
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]          # NCHW
```
```python
# training transform
T.Compose([T.Resize(96), T.ToTensor(),
           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
```

<details><summary>Hint 1</summary>

The ONNX model is fine (it works in the notebook). So the bug is in the *inputs*. List every step of the training transform and check it against the API, line by line.
</details>

<details><summary>Hint 2</summary>

What channel order does `cv2.imread` return?
</details>

<details><summary>Hint 3</summary>

Print the per-channel mean of the input tensor in the notebook and in the API for the same image. What ranges do you expect?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** three silent mismatches.
1. **BGR vs RGB**: OpenCV loads BGR, so red and blue are swapped (green/brown scenes read as frog or deer).
2. **Missing mean/std normalization**: inputs are in [0, 1] instead of about [−2.1, 2.6].
3. **Resize semantics**: `T.Resize(96)` on a non-square image resizes the *shorter side* to 96 and keeps the aspect ratio (training images are square, so it didn't matter there). `cv2.resize((96, 96))` squashes the image. The interpolation defaults also differ.

**Fix:** one shared preprocessing function used by training and the API (or bake the normalization into the exported graph):
```python
from PIL import Image, ImageOps
def preprocess(file):
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    x = eval_transform(img)             # the SAME torchvision eval transform as training
    return x.unsqueeze(0).numpy()
```
Add a **golden test**: 5 fixed images → expected logits (from the notebook) → assert the API matches within 1e-3.

**Lesson:** training/serving skew is the most common silent failure in ML serving. Nothing crashes; accuracy just disappears. Treat preprocessing as part of the model.
</details>

---

## Scenario 5 — "ONNX output differs from PyTorch!"

**Context:** after exporting the Day 2 model, a student writes a parity check.

**Symptoms**
```python
model = train(model)                       # returns after the last epoch
torch.onnx.export(model, torch.randn(1, 3, 96, 96), "m.onnx",
                  input_names=["input"], output_names=["logits"])

x = torch.randn(1, 3, 96, 96)
ref = model(x).detach().numpy()
sess = ort.InferenceSession("m.onnx")
out = sess.run(None, {"input": x.numpy()})[0]
print(np.abs(ref - out).max())            # 2.73 !!!
```
- Running the cell again gives a different `ref` each time, while `out` stays identical.
- Separately, the API fails for batch requests:
```text
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT :
Got invalid dimensions for input: input for the following indices
 index: 0 Got: 8 Expected: 1
```
- And one teammate gets `Unexpected input data type. Actual: (tensor(double)) , expected: (tensor(float))`.

<details><summary>Hint 1</summary>

Which of the two outputs is *non-deterministic*? What does that tell you about which side is wrong?
</details>

<details><summary>Hint 2</summary>

What mode is `model` in right after the training loop?
</details>

<details><summary>Hint 3</summary>

For the batch error, look at the `torch.onnx.export` arguments: how does ONNX know which dimensions may vary? For the dtype error, what dtype does `np.array(...) / 255` produce by default?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause**
1. **The reference is wrong, not the ONNX model.** The model is still in **train mode** after training, so the PyTorch reference has Dropout active and uses batch statistics in BN, and it is random on every call. The exporter defaults to `training=TrainingMode.EVAL`, so the ONNX graph has inference behaviour. The ONNX output is the correct one.
2. **Static shapes**: without `dynamic_axes`, the batch dimension is fixed at 1 (from the dummy input).
3. **dtype**: NumPy defaults to float64. The ONNX input is float32.

**Fix**
```python
model.eval()
torch.onnx.export(model, torch.randn(1, 3, 96, 96), "m.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17)
with torch.no_grad():
    ref = model(x).numpy()
np.testing.assert_allclose(ref, out, rtol=1e-3, atol=1e-5)
# API: x = x.astype(np.float32)
```

**Lesson:** a parity test is only as good as its reference. Expect **tiny** differences (about 1e-6 to 1e-5) from kernel and fusion differences. Differences at the 1e-1 level mean different computations. Always export from eval mode, declare dynamic axes, and match dtypes.
</details>

---

## Scenario 6 — The INT8 model is slower than FP32

**Context:** static PTQ of MobileNetV2 in PyTorch eager mode (Day 2). The student hit an error and "fixed" it.

**Symptoms**
```text
fp32  : 12.4 ms / image
int8  : 19.8 ms / image    (size 3.9 MB vs 14 MB, accuracy -0.8%)
```
The original error:
```text
NotImplementedError: Could not run 'aten::add.out' with arguments from the 'QuantizedCPU' backend.
```
The student's "fix" in the inverted-residual block:
```python
def forward(self, x):
    if self.use_res_connect:
        return self.quant(self.dequant(x) + self.dequant(self.conv(x)))   # dequant -> add in fp32 -> requant
    return self.conv(x)
```
Benchmark code:
```python
t = time.perf_counter(); model_int8(x); print(time.perf_counter() - t)   # single call
```
And no `fuse_modules` call appears anywhere in the notebook.

<details><summary>Hint 1</summary>

The model is 4× smaller, so the weights *are* int8. Where could time be spent *outside* the int8 convolutions?
</details>

<details><summary>Hint 2</summary>

Count how many quantize/dequantize conversions happen per forward pass. MobileNetV2 has 10 residual blocks.
</details>

<details><summary>Hint 3</summary>

Is a single timed call a fair benchmark? What about Conv → BN → ReLU as separate quantized ops?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause**
1. **Quant/dequant churn**: each residual add dequantizes two tensors, adds them in fp32, and requantizes. That is memory-bound conversion work repeated in every residual block.
2. **No fusion**: Conv, BN and ReLU run as separate ops, each with its own requantization step, instead of one fused `ConvReLU2d`.
3. **Bad benchmark**: one cold call includes one-time initialization and allocation. There is no warm-up and no repetition.
4. (Check too:) the quantization engine matches the CPU (`x86`/`fbgemm` on Intel/AMD, `qnnpack` on ARM such as Apple M-series or Oracle Ampere), and thread counts are equal in both runs.

**Fix**
- Use `torch.ao.nn.quantized.FloatFunctional()` (`self.skip_add.add(a, b)`) for residual adds, or simply use `torchvision.models.quantization.mobilenet_v2`, which already does this and provides `fuse_model()`.
- Fuse conv+bn(+relu) before `prepare`.
- Benchmark properly: warm up 10–20 runs, then time 100+, report median and p95, fix `torch.set_num_threads`.
- Alternative: ONNX Runtime static quantization (QDQ format), which handles fusion automatically.

**Lesson:** smaller is not automatically faster. The speed-up only appears when *the whole graph* stays in int8 with fused kernels on hardware that has fast int8 instructions. Always profile (`torch.profiler`) before concluding.
</details>

---

## Scenario 7 — Detector trains "perfectly" but finds no pedestrians

**Context:** Faster R-CNN MobileNetV3 fine-tuned on Penn-Fudan (Day 2 live demo), re-implemented by a student.

**Symptoms**
```text
Epoch 0: loss_classifier 0.0413  loss_box_reg 0.0021  loss_objectness 0.21  loss_rpn_box_reg 0.05
Epoch 3: loss_classifier 0.0009  loss_box_reg 0.0000  loss_objectness 0.04  loss_rpn_box_reg 0.02
```
```python
pred = model([img])[0]
print(pred["boxes"].shape)          # torch.Size([0, 4]) on every test image
```
```python
# dataset __getitem__
labels = torch.zeros((num_objs,), dtype=torch.int64)
# model setup
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=1)
```

<details><summary>Hint 1</summary>

`loss_box_reg` reached *exactly* 0. When does the box regression loss have nothing to regress?
</details>

<details><summary>Hint 2</summary>

In torchvision detection models, what does label `0` mean?
</details>

<details><summary>Hint 3</summary>

How many classes does the predictor need for "person" detection?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** in torchvision detectors, **class 0 is reserved for background**. Every pedestrian was labelled 0 (background) and the head has `num_classes=1` (background only). The model learned perfectly that "everything is background". The classifier loss is tiny, box regression has no foreground samples (so it is 0), and post-processing drops all background predictions, leaving zero boxes.

**Fix**
```python
labels = torch.ones((num_objs,), dtype=torch.int64)            # 1 = person
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # background + person
```
Also check boxes are `[x1, y1, x2, y2]` in pixels with x2 > x1 and y2 > y1 (torchvision raises "All bounding boxes should have positive height and width" otherwise).

**Lesson:** "loss went to zero" is a red flag, not a victory. Always **visualize predictions and ground truth** on a few images during training, and read the API contract (label conventions, box format).
</details>

---

## Scenario 8 — Container runs, but the host can't reach Flask

**Context:** Day 4 first Dockerfile.

**Symptoms**
```bash
$ docker run -p 5000:5000 cifar-api:v1
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
$ curl localhost:5000/health
curl: (56) Recv failure: Connection reset by peer
$ docker exec -it <id> curl localhost:5000/health
{"status": "ok"}
```

<details><summary>Hint 1</summary>

It works *inside* the container but not from the host. Where exactly does the request fail?
</details>

<details><summary>Hint 2</summary>

Read the Flask start-up line carefully. Whose "127.0.0.1" is that?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** Flask binds to the container's **loopback interface**. Docker's port forwarding delivers traffic to the container's `eth0` address, where nothing is listening, so the connection is reset.

**Fix:** `app.run(host="0.0.0.0", port=5000)` or, better for production, `CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "app:app"]`.

**Lesson:** every container has its own network namespace, and `localhost` means "this container". Servers in containers must bind to `0.0.0.0`. `EXPOSE` documents the port; `-p` publishes it.
</details>

---

## Scenario 9 — Streamlit can't reach the API in docker-compose

**Symptoms**
```text
streamlit-1  | requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=5000):
streamlit-1  | Max retries exceeded with url: /predict (Caused by NewConnectionError(
streamlit-1  | '<urllib3.connection.HTTPConnection object>: Failed to establish a new connection:
streamlit-1  | [Errno 111] Connection refused'))
```
```yaml
services:
  api:
    build: ./api
    ports: ["5000:5000"]
  ui:
    build: ./ui
    ports: ["8501:8501"]
    depends_on: [api]
```
```python
API_URL = "http://localhost:5000"
```
"But `curl localhost:5000/health` works from my laptop!"

<details><summary>Hint 1</summary>

The curl runs on the *host*. Where does the Streamlit code run?
</details>

<details><summary>Hint 2</summary>

How do containers on the same compose network find each other?
</details>

<details><summary>Hint 3</summary>

After fixing the host name, the first request sometimes still fails right after `docker compose up`. Why doesn't `depends_on` prevent that?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** inside the `ui` container, `localhost` is the UI container itself. Compose puts services on a shared network with **DNS by service name**. Also, `depends_on` only orders container *start*; it doesn't wait for the API to be *ready* (model loaded).

**Fix**
```yaml
  api:
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 5s
      retries: 10
  ui:
    environment:
      - API_URL=http://api:5000
    depends_on:
      api:
        condition: service_healthy
```
```python
API_URL = os.getenv("API_URL", "http://localhost:5000")
```
Add retries with backoff in the client as well.

**Lesson:** never hard-code service addresses; configure them per environment. Understand what "localhost" means in each network namespace. Start-up order is not readiness.
</details>

---

## Scenario 10 — `ImagePullBackOff` on minikube/kind

**Symptoms**
```bash
$ docker build -t cifar-api:latest .
$ kubectl apply -f deployment.yaml
$ kubectl get pods
NAME                         READY   STATUS             RESTARTS   AGE
cifar-api-7d9c8b6f5d-x2x7k   0/1     ImagePullBackOff   0          2m
$ kubectl describe pod cifar-api-7d9c8b6f5d-x2x7k
  Warning  Failed   kubelet  Failed to pull image "cifar-api:latest": ... pull access denied,
                             repository does not exist or may require authorization
```

<details><summary>Hint 1</summary>

Where was the image built, and where does the kubelet look for it?
</details>

<details><summary>Hint 2</summary>

What `imagePullPolicy` does Kubernetes use by default for a `:latest` tag?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** the image exists only in your **host's Docker daemon**. minikube and kind nodes run their own container runtime, and there is no such image on Docker Hub. The `:latest` tag also implies `imagePullPolicy: Always`, so Kubernetes tries the registry even if a local copy exists on the node.

**Fix** (any of these)
- minikube: `minikube image load cifar-api:v1`, or build inside minikube: `eval $(minikube docker-env)` then `docker build ...`.
- kind: `kind load docker-image cifar-api:v1`.
- Push to a registry (Docker Hub, GHCR) and add an `imagePullSecret` if it is private.
- Use an explicit tag (`:v1`, or the git SHA) and `imagePullPolicy: IfNotPresent`.

Other `ImagePullBackOff` causes to check: typos in the image name or tag, a private registry without credentials, Docker Hub rate limits, an architecture mismatch (it pulls, then fails with `exec format error`).

**Lesson:** "the cluster" is a different machine from your laptop even when it runs on your laptop. Use immutable tags. `kubectl describe pod` and the Events section are the first place to look.
</details>

---

## Scenario 11 — `CrashLoopBackOff` with exit code 137

**Symptoms**
```bash
$ kubectl get pods
cifar-api-5f6d...   0/1   CrashLoopBackOff   5 (40s ago)   6m
$ kubectl describe pod cifar-api-5f6d...
    Last State:     Terminated
      Reason:       OOMKilled
      Exit Code:    137
    Limits:
      cpu:     500m
      memory:  512Mi
$ kubectl logs cifar-api-5f6d... --previous
[INFO] Booting worker with pid: 8
[INFO] Booting worker with pid: 9
[INFO] Booting worker with pid: 10
[INFO] Booting worker with pid: 11
```
The image is the **torch-based** API: `gunicorn -w 4 app:app`, and each worker runs `torch.load(...)` at import.

<details><summary>Hint 1</summary>

Exit code 137 = 128 + 9. Which signal is 9, and who sent it?
</details>

<details><summary>Hint 2</summary>

How many copies of PyTorch and the model are in memory? Roughly how much RAM does `import torch` plus a model use per process?
</details>

<details><summary>Hint 3</summary>

Run it with `docker run -m 512m` locally and watch `docker stats`. When does memory peak?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** SIGKILL from the kernel OOM killer when the cgroup memory limit was exceeded. Each of the 4 gunicorn workers imports torch (often a few hundred MB RSS per process) and loads its own model copy. The total goes well past 512 Mi, so the container is killed, restarted, and killed again, which is CrashLoopBackOff.

**Fix** (combine)
- Serve the **ONNX Runtime** model instead of torch: much smaller footprint and image.
- Match workers to the CPU limit: a 500m CPU limit doesn't justify 4 workers. Use `-w 1` or `-w 2` with ORT threads = 1.
- `gunicorn --preload` to load once and share read-only pages (partial sharing through copy-on-write).
- Measure the real peak (`docker stats`, `kubectl top pod`) and set `requests.memory = limits.memory` about 30% above it.

**Lesson:** memory limits are hard walls. CPU over its limit gets throttled; memory over its limit gets killed. Size workers to the resources, and measure peak memory, not the model file size.
</details>

---

## Scenario 12 — Pods restart forever during start-up

**Symptoms**
```yaml
livenessProbe:
  httpGet: { path: /health, port: 5000 }
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 3
readinessProbe:
  httpGet: { path: /health, port: 5000 }
  initialDelaySeconds: 5
  periodSeconds: 5
```
```text
Events:
  Warning  Unhealthy  kubelet  Liveness probe failed: Get "http://10.244.0.12:5000/health":
                               dial tcp 10.244.0.12:5000: connect: connection refused
  Normal   Killing    kubelet  Container api failed liveness probe, will be restarted
```
The app logs show "Loading model…" and never reach "Model loaded in 21.4s". Locally, the container starts fine.

<details><summary>Hint 1</summary>

Add up `initialDelaySeconds + periodSeconds × failureThreshold`. Compare with the load time.
</details>

<details><summary>Hint 2</summary>

Why is the load slower in the cluster than on your laptop? (Look at the CPU limit.)
</details>

<details><summary>Hint 3</summary>

Should liveness and readiness ask the same question? (See Q7.2.)
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** liveness starts failing after about 5 + 5 × 3 = **20 s**, but model loading takes about 21 s, and more under a CPU limit of a few hundred millicores. The kubelet kills the container just before it finishes, and it happens again on every restart (with growing back-off). The same endpoint is used for both "am I alive" and "am I ready".

**Fix**
```yaml
startupProbe:                     # gives up to 30 × 2 = 60 s to start; other probes wait for it
  httpGet: { path: /health/live, port: 5000 }
  periodSeconds: 2
  failureThreshold: 30
livenessProbe:                    # cheap: "the process responds"
  httpGet: { path: /health/live, port: 5000 }
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3
readinessProbe:                   # "model loaded, can serve" -> returns 503 until then
  httpGet: { path: /health/ready, port: 5000 }
  periodSeconds: 5
```
Also speed up start-up: a smaller model (int8 ONNX), a lighter runtime, and a warm-up inference after loading.

**Lesson:** use a startupProbe for slow initialization. Keep liveness cheap and dependency-free. Readiness means "can serve now". Probe budgets must be tested under the *cluster's* CPU limits.
</details>

---

## Scenario 13 — Streamlit in k8s: "Failed to resolve 'api'"

**Symptoms**
```text
requests.exceptions.ConnectionError: HTTPConnectionPool(host='api', port=5000): Max retries exceeded
(Caused by NameResolutionError("Failed to resolve 'api' ([Errno -2] Name or service not known)"))
```
```yaml
# configmap.yaml
data:
  API_URL: "http://api:5000"          # copied from docker-compose
---
# api-service.yaml
apiVersion: v1
kind: Service
metadata: { name: cifar-api-svc }
spec:
  selector: { app: cifar-api }
  ports: [{ port: 80, targetPort: 5000 }]
---
# api deployment pod template labels
labels: { app: cifar-api-v2 }
```
After fixing the name to `http://cifar-api-svc:5000`, the error changes to a **timeout**. After changing the port to 80, it becomes **connection refused**.

<details><summary>Hint 1</summary>

In Kubernetes, which object provides the DNS name, and what is it called here?
</details>

<details><summary>Hint 2</summary>

Which port does a client use: `port` or `targetPort`?
</details>

<details><summary>Hint 3</summary>

Run `kubectl get endpoints cifar-api-svc`. What do you see?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** three stacked bugs.
1. **Wrong DNS name**: `api` was the compose service name. In k8s the name comes from the **Service**: `cifar-api-svc` (or `cifar-api-svc.<namespace>.svc.cluster.local`).
2. **Wrong port**: clients call the Service `port` (80). `targetPort` (5000) is the container port.
3. **Selector mismatch**: the Service selects `app: cifar-api`, but the pods are labelled `app: cifar-api-v2`, so **endpoints are `<none>`** and nothing is behind the Service.

**Fix:** `API_URL: "http://cifar-api-svc"` (port 80). Align the Service selector with the pod labels. Restart the UI pods after the ConfigMap change (env vars are read at start: `kubectl rollout restart deploy/ui`).

**Debug toolkit:** `kubectl get svc,endpoints`, `kubectl exec -it <ui-pod> -- python -c "import socket; print(socket.gethostbyname('cifar-api-svc'))"`, `kubectl run tmp --rm -it --image=curlimages/curl -- sh`.

**Lesson:** debug networking layer by layer: DNS → Service → endpoints (selector) → port mapping → container bind address. And remember that ConfigMap changes don't reach running pods' environment variables.
</details>

---

## Scenario 14 — HPA shows `<unknown>/70%` and never scales

**Symptoms**
```bash
$ kubectl get hpa
NAME        REFERENCE              TARGETS         MINPODS   MAXPODS   REPLICAS
cifar-api   Deployment/cifar-api   <unknown>/70%   1         5         1
$ kubectl describe hpa cifar-api
  Warning  FailedGetResourceMetric  horizontal-pod-autoscaler
           failed to get cpu utilization: unable to get metrics for resource cpu:
           unable to fetch metrics from resource metrics API
```
After the team enables something, a new warning appears:
```text
failed to get cpu utilization: missing request for cpu in container "api"
```
A load test with `hey -z 60s -c 50` pushes p95 latency to 4 s, still with 1 replica.

<details><summary>Hint 1</summary>

Where does the HPA get CPU numbers from? Does `kubectl top pods` work?
</details>

<details><summary>Hint 2</summary>

"70% utilization" means 70% *of what*?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause**
1. **metrics-server not installed** (minikube: disabled by default; kind: not installed; k3s: included). The HPA has no CPU data.
2. **No CPU requests** on the container. Utilization is computed as usage ÷ *request*; without a request the percentage is undefined.

**Fix**
- `minikube addons enable metrics-server` (kind: install metrics-server, often with `--kubelet-insecure-tls`).
- Add `resources.requests.cpu: 250m` (and a memory request and limit). Check that `kubectl top pods` works.
- Re-run the load test and watch `kubectl get hpa -w`. Expect a delay of tens of seconds before scaling (see Q7.4).

**Lesson:** autoscaling is a control loop. It needs a sensor (metrics), a reference (requests) and time to act. Test it under load *before* you rely on it.
</details>

---

## Scenario 15 — Render demo times out on the first request

**Context:** Day 4. The API is deployed on Render's free tier, and the Streamlit UI (on HF Spaces) calls it.

**Symptoms**
- During the lecturer's demo, the first click fails. The second click, 40 s later, works instantly.
```text
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='cifar-api.onrender.com', port=443):
Read timed out. (read timeout=10)
```
- The Render logs show a fresh container start and "Model loaded" roughly 45 s after the request arrived.
- The image is 2.1 GB (a torch base).

<details><summary>Hint 1</summary>

What does the free tier do with services that get no traffic for a while?
</details>

<details><summary>Hint 2</summary>

List everything that happens between "request arrives at a sleeping service" and "first response". Which parts do *you* control?
</details>

<details><summary>Root cause, fix, lesson</summary>

**Root cause:** Render's free web services **spin down after about 15 minutes without traffic**. The next request triggers a **cold start**: schedule, pull or unpack the image (2.1 GB is slow), boot Python, import torch, load the model. That takes more than the client's 10 s timeout.

**Fix**
- Shrink the cold start: ONNX Runtime image (about 300 MB), lazy imports, int8 model, load it at start-up.
- Client: longer timeout for the first call, retries with backoff, a "Waking up the model server…" spinner, and a `/health` warm-up call when the UI loads.
- For real users: a paid always-on instance or `min instances ≥ 1` elsewhere. (Keep-alive pings on a free tier may breach the provider's terms. Discuss.)
- Before a live demo: warm the service up 2 minutes before you present.

**Lesson:** free tiers trade money for latency and availability. Design for cold starts (image size, start-up time, client timeouts, UX), and always rehearse the demo path end-to-end.
</details>

---

## Wrap-up questions for this block

1. Which scenarios produced **no error at all**, only wrong results? (3, 4, 5-reference, 7.) Why are those the most dangerous?
2. Which **automated test** would have caught each bug before users did?
3. What is the order of layers you now check when "the prediction is wrong"? When "the service is down"?
