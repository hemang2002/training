# Curated Reading and Reference List
### PyTorch → Transfer Learning → Model Optimization → Detection & GANs → Flask/Streamlit → Docker → Kubernetes → Free Cloud → MLOps

**Links verified: 2026-09-26.** Every URL returned HTTP 200 on that date, and redirects were resolved to their final URL.
Versions seen on the official docs that day: **PyTorch 2.14**, **torchvision 0.29**, **torchao 0.17**, **Flask 3.1.x**, **ONNX 1.24**.

**Legend**
- **Type:** Docs = official docs · Tut = tutorial · Paper · Blog · Video · Course · Tool
- **Level:** B = beginner · I = intermediate · A = advanced
- **Time:** a rough estimate of reading or working-through time, not including running code unless it says so.
- **(S)** means it's a good one to **share directly with students**. Unmarked items are mainly for the trainer's own depth.

> **Two things changed recently. Adjust your slides.**
> 1. **Quantization in PyTorch has moved to `torchao`.** The `torch.ao.quantization` page now says it's deprecated: *"We are centralizing all quantization related development to torchao."* Teach the torchao / PT2E APIs; treat eager-mode `torch.ao.quantization` as legacy.
> 2. **The old "PyTorch Flask REST API" tutorial is gone.** Its URL now redirects away. Use the Flask docs (section 7) instead.

---

## 1. Foundations and intuition (warm-up)

1. **3Blue1Brown: Neural Networks (video series)** (S)
   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
   Video · B · ~2.5 h
   *Why:* the best visual intuition for neurons, gradient descent and backprop. Assign chapters 1–4 before the course starts.

2. **Andrej Karpathy: "The spelled-out intro to neural networks and backpropagation: building micrograd"**
   https://www.youtube.com/watch?v=VMj-3S1tku0
   Video · B/I · ~2.5 h
   *Why:* builds autograd from scratch. After it you can explain exactly what `loss.backward()` does.

3. **CS231n: Deep Learning for Computer Vision (course notes)** (S)
   https://cs231n.github.io/
   Course · I · browse; ~1 h per module
   *Why:* Stanford's classic notes on CNNs, optimisation and training. The best written reference for the vision half of the course.

4. **Dive into Deep Learning (d2l.ai)** (S)
   https://d2l.ai/
   Course/book · B–A · reference
   *Why:* free interactive textbook with PyTorch code for every concept. Good for students who want "the textbook version."

5. **Deep Learning (Goodfellow, Bengio, Courville): online book**
   https://www.deeplearningbook.org/
   Book · A · reference
   *Why:* the rigorous theory reference (regularisation, optimisation, generative models). Use it to answer "why" questions from MSc students.

6. **fast.ai: Practical Deep Learning for Coders** (S)
   https://course.fast.ai/
   Course · B/I · ~20 h total
   *Why:* top-down, code-first teaching. Worth watching lesson 1 for pedagogy ideas and transfer-learning-first framing.

7. **Andrej Karpathy: "A Recipe for Training Neural Networks"** (S) ★
   https://karpathy.github.io/2019/04/25/recipe/
   Blog · I · 25 min
   *Why:* a practical debugging discipline: overfit one batch, verify the loss at init, add complexity slowly. Make it required reading.

8. **CS231n: Neural Networks Part 3 (learning and evaluation)**
   https://cs231n.github.io/neural-networks-3/
   Course notes · I · 40 min
   *Why:* gradient checks, sanity checks, babysitting the learning process, and learning-rate schedules. The practical companion to #7.

9. **Distill: Feature Visualization**
   https://distill.pub/2017/feature-visualization/
   Blog (peer-reviewed) · I · 30 min
   *Why:* beautiful interactive article on what CNN layers learn. It motivates why pretrained features transfer (section 3).

---

## 2. PyTorch training fundamentals

10. **PyTorch: Learn the Basics** (S) ★
    https://docs.pytorch.org/tutorials/beginner/basics/intro.html
    Tut · B · 2–3 h with code
    *Why:* the official end-to-end path (tensors → data → model → autograd → optimisation → save). Mirror its structure in Day 1.

11. **Datasets & DataLoaders** (S)
    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    Tut · B · 30 min
    *Why:* custom `Dataset`, `DataLoader`, batching and workers. Students need this for every later lab, including detection.

12. **Build the Neural Network** (S)
    https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    Tut · B · 20 min
    *Why:* `nn.Module`, `forward`, layers and parameters, in the idiomatic style.

13. **Optimizing Model Parameters (the training loop)** (S)
    https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    Tut · B · 25 min
    *Why:* the canonical train/test loop, with `zero_grad` / `backward` / `step`. Keep it as a reference slide.

14. **Introduction to PyTorch: YouTube Series**
    https://docs.pytorch.org/tutorials/beginner/introyt/introyt_index.html
    Video + Tut · B · ~3 h
    *Why:* official video walkthroughs of the same material. Good for students who prefer video.

15. **Saving and Loading Models** (S)
    https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    Tut · B · 20 min
    *Why:* `state_dict` vs whole-model saving, and checkpoints for resuming. Essential before serving a model in Flask.

16. **Automatic Mixed Precision (AMP) recipe**
    https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    Tut · I · 20 min
    *Why:* `torch.autocast` + `GradScaler` for faster Colab GPU training. A good bridge to "precision" before quantization.

17. **Performance Tuning Guide**
    https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    Docs · I/A · 30 min
    *Why:* checklist of practical speedups (DataLoader workers, pinned memory, `no_grad`, channels-last, and so on).

18. **Reproducibility (randomness notes)** (S)
    https://docs.pytorch.org/docs/stable/notes/randomness.html
    Docs · I · 10 min
    *Why:* seeding and deterministic algorithms. MSc students must report reproducible results.

19. **Introduction to `torch.compile`**
    https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    Tut · I/A · 30 min
    *Why:* PyTorch 2.x graph compilation. Background for PT2E quantization and export.

20. **Visualizing Models, Data and Training with TensorBoard** (S)
    https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    Tut · B/I · 30 min
    *Why:* logging curves and images. A first step toward experiment tracking (MLOps, section 11).

---

## 3. Transfer learning

21. **Transfer Learning for Computer Vision Tutorial** (S) ★
    https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    Tut · B/I · 45 min + run
    *Why:* fine-tuning vs fixed feature extractor on ResNet-18. The core lab template.

22. **CS231n: Transfer Learning notes** (S)
    https://cs231n.github.io/transfer-learning/
    Course notes · B/I · 15 min
    *Why:* the decision matrix (small vs large dataset, similar vs different domain). Gives you when to freeze vs fine-tune.

23. **d2l.ai: Fine-Tuning**
    https://d2l.ai/chapter_computer-vision/fine-tuning.html
    Book · I · 30 min
    *Why:* explains using a lower learning rate for pretrained layers and a higher one for the new head, with code.

24. **TorchVision: Models and pre-trained weights** (S)
    https://docs.pytorch.org/vision/stable/models.html
    Docs · B/I · 20 min
    *Why:* the multi-weight API (`weights=ResNet50_Weights.DEFAULT`, `weights.transforms()`) and accuracy/size tables for choosing a backbone.

25. **TorchVision: Transforms (v2)**
    https://docs.pytorch.org/vision/stable/transforms.html
    Docs · I · 25 min
    *Why:* v2 transforms handle images **and bounding boxes** together, which the detection lab needs.

26. **Yosinski et al. (2014): "How transferable are features in deep neural networks?"**
    https://arxiv.org/abs/1411.1792
    Paper · I/A · 1 h
    *Why:* the empirical basis for transfer learning (general early layers vs specific late layers). Cite it in lectures.

27. **He et al. (2015): Deep Residual Learning (ResNet)**
    https://arxiv.org/abs/1512.03385
    Paper · I · 1 h
    *Why:* the default backbone in every lab (classification and Faster R-CNN). Know skip connections cold.

28. **Sandler et al. (2018): MobileNetV2**
    https://arxiv.org/abs/1801.04381
    Paper · I · 1 h
    *Why:* inverted residuals and depthwise-separable convs. The go-to small backbone for edge deployment and free-tier serving.

---

## 4. Model optimization: quantization, pruning, distillation, ONNX

### 4a. Overviews
29. **HF blog: "Efficient Deep Learning: A Comprehensive Overview of Optimization Techniques"**
    https://huggingface.co/blog/Isayoften/optimization-rush
    Blog · I · 45 min
    *Why:* one-stop survey (quantization, pruning, distillation, mixed precision). A good map before going deep.

30. **Lilian Weng: "Large Transformer Model Inference Optimization"**
    https://lilianweng.github.io/posts/2023-01-10-inference-optimization/
    Blog · A · 45 min
    *Why:* clear taxonomy of quantization, pruning, sparsity and distillation. The concepts apply to CNNs too.

### 4b. Quantization
31. **PyTorch Quantization page (deprecation notice → torchao)** ★
    https://docs.pytorch.org/docs/stable/quantization.html
    Docs · I · 10 min
    *Why:* read the migration notes (eager → `quantize_`, FX → PT2E) so your notebooks don't rely on APIs that are being removed.

32. **torchao documentation** ★
    https://docs.pytorch.org/ao/stable/index.html
    Docs · I/A · 45 min
    *Why:* the current home of PyTorch quantization and sparsity (int8/int4 weight-only, dynamic, QAT).

33. **torchao: PT2E Quantization**
    https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html
    Docs · A · 45 min
    *Why:* export-based static post-training quantization (PTQ) and QAT flow (`prepare_pt2e` / `convert_pt2e`). The replacement for FX graph-mode quantization.

34. **torchao: Workflows**
    https://docs.pytorch.org/ao/stable/workflows/index.html
    Docs · I · 20 min
    *Why:* which quantization recipe to use for which hardware or model. Helps you pick a demo that actually speeds up on CPU.

35. **PyTorch blog: "PyTorch Native Architecture Optimization: torchao"**
    https://pytorch.org/blog/pytorch-native-architecture-optimization/
    Blog · I · 15 min
    *Why:* the motivation and benchmarks behind torchao. Good context slide.

36. **PyTorch blog: "Practical Quantization in PyTorch"** (S)
    https://pytorch.org/blog/quantization-in-practice/
    Blog · I · 30 min
    *Why:* still the clearest conceptual explainer of dynamic vs static vs QAT, calibration and per-channel. The code uses the older API, so pair it with #32.

37. **Jacob et al. (2017): "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"** ★
    https://arxiv.org/abs/1712.05877
    Paper · A · 1.5 h
    *Why:* the foundational scale/zero-point affine scheme and fake-quant QAT. Every framework implements this.

38. **Wu et al. (NVIDIA, 2020): "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation"**
    https://arxiv.org/abs/2004.09602
    Paper · A · 1.5 h
    *Why:* practical recipes (per-channel weights, calibration methods, which layers to keep in FP) with accuracy tables.

39. **Krishnamoorthi (2018): "Quantizing deep convolutional networks for efficient inference: A whitepaper"**
    https://arxiv.org/abs/1806.08342
    Paper · I/A · 1 h
    *Why:* very readable whitepaper on PTQ vs QAT for CNNs, with MobileNet results. Ideal for MSc reading.

40. **Gholami et al. (2021): "A Survey of Quantization Methods for Efficient Neural Network Inference"**
    https://arxiv.org/abs/2103.13630
    Paper · A · 2 h (skim)
    *Why:* comprehensive survey. Useful for the "further reading" slide and student literature reviews.

### 4c. Pruning, sparsity, compression
41. **PyTorch: Pruning Tutorial** (S) ★
    https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html
    Tut · I · 40 min
    *Why:* `torch.nn.utils.prune` (unstructured/structured, global, making pruning permanent). Also demonstrates that unstructured sparsity doesn't shrink files or speed up dense kernels by itself.

42. **Han et al. (2015): "Deep Compression"**
    https://arxiv.org/abs/1510.00149
    Paper · A · 1 h
    *Why:* the pruning → quantization → Huffman pipeline (35–49× compression). A classic lecture case study.

43. **Frankle & Carbin (2018): "The Lottery Ticket Hypothesis"**
    https://arxiv.org/abs/1803.03635
    Paper · A · 1 h
    *Why:* why sparse subnetworks exist and can be trained. Great discussion paper for MSc students.

### 4d. Knowledge distillation
44. **PyTorch: Knowledge Distillation Tutorial** (S) ★
    https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
    Tut · I · 45 min + run
    *Why:* teacher/student on CIFAR-10 with soft targets, temperature and a feature-matching loss. A lab-ready template.

45. **Hinton, Vinyals, Dean (2015): "Distilling the Knowledge in a Neural Network"**
    https://arxiv.org/abs/1503.02531
    Paper · I/A · 45 min
    *Why:* the original soft-target/temperature idea ("dark knowledge"). Short and very readable.

### 4e. ONNX and ONNX Runtime
46. **PyTorch: Introduction to ONNX** (S)
    https://docs.pytorch.org/tutorials/beginner/onnx/intro_onnx.html
    Tut · B · 15 min
    *Why:* what ONNX is and why you'd export. Sets up the next item.

47. **PyTorch: Export a PyTorch model to ONNX** (S) ★
    https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
    Tut · I · 30 min
    *Why:* the current `torch.onnx.export(..., dynamo=True)` path and checking outputs with ONNX Runtime.

48. **torch.onnx API docs**
    https://docs.pytorch.org/docs/stable/onnx.html
    Docs · I/A · reference
    *Why:* dynamic shapes, opsets, and troubleshooting export failures (common with detection models).

49. **ONNX: Introduction (onnx.ai)**
    https://onnx.ai/onnx/intro/
    Docs · I · 30 min
    *Why:* the graph/opset/IR concepts behind the file format. Helps when reading models in Netron.

50. **ONNX Runtime: Python get-started** (S)
    https://onnxruntime.ai/docs/get-started/with-python.html
    Docs · B · 20 min
    *Why:* `InferenceSession` basics. This is what the Flask API will run on free-tier CPUs.

51. **ONNX Runtime: Export PyTorch model tutorial**
    https://onnxruntime.ai/docs/tutorials/export-pytorch-model.html
    Tut · I · 20 min
    *Why:* the ORT side of the export story, with inference examples.

52. **ONNX Runtime: Quantize ONNX models** ★
    https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
    Docs · I/A · 45 min
    *Why:* dynamic vs static quantization, QDQ vs QOperator formats, calibration, and the pre-processing step. The easiest way to get a real CPU speedup for the deploy demo.

53. **ONNX Runtime: Graph optimizations**
    https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
    Docs · I · 20 min
    *Why:* constant folding and operator fusion levels, plus how to save the optimised model.

54. **Netron (model viewer)** (S)
    https://netron.app/
    Tool · B · 5 min
    *Why:* drag in `.onnx` / `.pt` files to visualise graphs. Great live in class, and for spotting QDQ nodes after quantization.

55. **Hugging Face Optimum (docs)**
    https://huggingface.co/docs/optimum/index
    Docs · I · 20 min
    *Why:* high-level export and quantization tooling (ONNX Runtime, OpenVINO and others). Good to mention as the industry shortcut.

56. **NVIDIA TensorRT documentation**
    https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
    Docs · A · reference
    *Why:* where ONNX goes for GPU production inference. A "what's next" pointer, not needed for labs.

---

## 5. Object detection (Faster R-CNN, torchvision fine-tuning)

57. **TorchVision Object Detection Finetuning Tutorial** (S) ★
    https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    Tut · I · 1 h + run
    *Why:* Penn-Fudan fine-tuning of Faster R-CNN / Mask R-CNN. Covers the target dict format (`boxes`, `labels`), replacing the box predictor, and the training loop. The core lab.

58. **TorchVision: Faster R-CNN model builders**
    https://docs.pytorch.org/vision/stable/models/faster_rcnn.html
    Docs · I · 15 min
    *Why:* available variants (ResNet-50 FPN v1/v2, MobileNetV3) with box mAP numbers. Choose MobileNetV3 for CPU/free-tier demos.

59. **TorchVision: `fasterrcnn_resnet50_fpn_v2`**
    https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
    Docs · I · 10 min
    *Why:* exact input/output contract (train vs eval mode returns losses vs detections), which you need for serving.

60. **Lilian Weng: "Object Detection for Dummies Part 1"** (S)
    https://lilianweng.github.io/posts/2017-10-29-object-recognition-part-1/
    Blog · B · 25 min
    *Why:* pre-deep-learning foundations (HOG, selective search) that make region proposals intuitive.

61. **Lilian Weng: "Object Detection for Dummies Part 3: R-CNN Family"** (S) ★
    https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/
    Blog · I · 40 min
    *Why:* the best single read on R-CNN → Fast → Faster → Mask R-CNN, with RPN, anchors and RoI pooling explained visually.

62. **d2l.ai: Region-based CNNs (R-CNNs)** (S)
    https://d2l.ai/chapter_computer-vision/rcnn.html
    Book · I · 30 min
    *Why:* textbook treatment with RoI pooling illustrated. Good student handout.

63. **Girshick et al. (2013): R-CNN**
    https://arxiv.org/abs/1311.2524
    Paper · I/A · 1 h
    *Why:* historical starting point. Read the intro and method only.

64. **Girshick (2015): Fast R-CNN**
    https://arxiv.org/abs/1504.08083
    Paper · I/A · 45 min
    *Why:* RoI pooling and the multi-task loss. The bridge to Faster R-CNN.

65. **Ren et al. (2015): Faster R-CNN** ★
    https://arxiv.org/abs/1506.01497
    Paper · A · 1.5 h
    *Why:* the Region Proposal Network and anchors. Be able to draw the architecture on a whiteboard.

66. **Lin et al. (2016): Feature Pyramid Networks (FPN)**
    https://arxiv.org/abs/1612.03144
    Paper · A · 1 h
    *Why:* the "FPN" in `fasterrcnn_resnet50_fpn`: multi-scale features for small objects.

67. **Lin et al. (2014): Microsoft COCO dataset paper**
    https://arxiv.org/abs/1405.0312
    Paper · I · 30 min (skim)
    *Why:* the dataset behind torchvision's pretrained detectors and the benchmark students will cite.

68. **COCO: Detection evaluation (mAP@[.5:.95], AP_small, …)** (S)
    https://cocodataset.org/#detection-eval
    Docs · I · 20 min
    *Why:* the official definition of the metrics students will report. Clears up IoU thresholds and AP vs AR.

---

## 6. Generative Adversarial Networks (DCGAN)

69. **PyTorch: DCGAN Tutorial** (S) ★
    https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    Tut · I · 1 h + run
    *Why:* complete DCGAN on CelebA (weight init, G/D loops, loss curves). The core lab.

70. **Goodfellow et al. (2014): "Generative Adversarial Networks"** ★
    https://arxiv.org/abs/1406.2661
    Paper · I/A · 1 h
    *Why:* the minimax game and the optimal-discriminator proof. Present the value function from here.

71. **Radford, Metz, Chintala (2015): DCGAN paper** ★
    https://arxiv.org/abs/1511.06434
    Paper · I · 45 min
    *Why:* the architecture guidelines (strided convs, BatchNorm, ReLU/LeakyReLU, no FC layers) that the PyTorch tutorial implements.

72. **Goodfellow (2016): NIPS 2016 Tutorial: Generative Adversarial Networks**
    https://arxiv.org/abs/1701.00160
    Paper (tutorial) · I/A · 2 h
    *Why:* the inventor's own long-form explanation, including mode collapse and tips. The best deep-dive for the trainer.

73. **Google ML: GAN course** (S)
    https://developers.google.com/machine-learning/gan
    Course · B · ~1 h
    *Why:* short, friendly intro (generator/discriminator, loss functions, common problems). Good pre-reading for students.

74. **Lilian Weng: "From GAN to WGAN"**
    https://lilianweng.github.io/posts/2017-08-20-gan/
    Blog · A · 45 min
    *Why:* the math of why GAN training is unstable (JS divergence) and how Wasserstein loss helps. Good for strong MSc students.

75. **soumith/ganhacks: "How to Train a GAN?"** (S)
    https://github.com/soumith/ganhacks
    Blog/notes · I · 15 min
    *Why:* practical tricks (label smoothing, noisy labels, Adam, avoid sparse gradients). Useful when student GANs collapse.

76. **Distill: "Deconvolution and Checkerboard Artifacts"**
    https://distill.pub/2016/deconv-checkerboard/
    Blog (peer-reviewed) · I · 20 min
    *Why:* explains the checkerboard patterns in DCGAN outputs, and why resize-then-conv helps.

77. **Distill: "Open Questions about Generative Adversarial Networks"**
    https://distill.pub/2019/gan-open-problems/
    Blog (peer-reviewed) · A · 30 min
    *Why:* research-level discussion questions for MSc seminars.

78. **Karras et al. (2017): Progressive Growing of GANs**
    https://arxiv.org/abs/1710.10196
    Paper · A · 1 h
    *Why:* shows how GANs got from DCGAN-quality to high-res faces. A "where the field went next" slide.

---

## 7. Serving: Flask + Streamlit

79. **Flask: Quickstart** (S) ★
    https://flask.palletsprojects.com/en/stable/quickstart/
    Docs · B · 45 min
    *Why:* routing, `request`, JSON responses, file uploads. Everything needed for a `/predict` endpoint. Note the warning that the dev server isn't for production.

80. **Flask: Tutorial (Flaskr)**
    https://flask.palletsprojects.com/en/stable/tutorial/
    Tut · B/I · 2 h
    *Why:* app factory, blueprints, config and tests. Use it to structure a clean inference API.

81. **Flask: Deploying to Production** (S)
    https://flask.palletsprojects.com/en/stable/deploying/
    Docs · I · 15 min
    *Why:* why you need a WSGI server in containers.

82. **Flask: Gunicorn** (S)
    https://flask.palletsprojects.com/en/stable/deploying/gunicorn/
    Docs · I · 10 min
    *Why:* the exact `gunicorn -w … app:app` command for the Dockerfile `CMD`. Size workers carefully on 512 MB free tiers.

83. **Streamlit: Get started** (S) ★
    https://docs.streamlit.io/get-started
    Docs · B · 45 min
    *Why:* install, first app, and the core concepts.

84. **Streamlit: Run your app (execution model)**
    https://docs.streamlit.io/develop/concepts/architecture/run-your-app
    Docs · B · 10 min
    *Why:* the whole script reruns on every interaction. This is why caching matters.

85. **Streamlit: Caching overview** (S) ★
    https://docs.streamlit.io/develop/concepts/architecture/caching
    Docs · I · 25 min
    *Why:* `st.cache_resource` for loading the model once and `st.cache_data` for data. The #1 performance bug in student apps.

86. **Streamlit: Session State**
    https://docs.streamlit.io/develop/concepts/architecture/session-state
    Docs · I · 20 min
    *Why:* keep state across reruns (history, uploaded images).

87. **Streamlit: `st.file_uploader`** (S)
    https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
    Docs · B · 10 min
    *Why:* image upload for classification or detection demos. Pass the bytes on to the Flask API.

88. **Streamlit: Deploy using Docker** (S) ★
    https://docs.streamlit.io/deploy/tutorials/docker
    Tut · I · 30 min
    *Why:* the official Dockerfile, `--server.port` / `--server.address=0.0.0.0`, and a healthcheck (`/_stcore/health`).

---

## 8. Docker

89. **Docker: What is Docker? (overview)** (S)
    https://docs.docker.com/get-started/docker-overview/
    Docs · B · 20 min
    *Why:* images vs containers, the daemon, registries. Vocabulary for Day 3.

90. **Docker: Get started** (S) ★
    https://docs.docker.com/get-started/
    Tut · B · 2–3 h
    *Why:* the official hands-on path (build, run, share, volumes, compose). Students can self-study it.

91. **Docker: Python language-specific guide** (S)
    https://docs.docker.com/guides/python/
    Tut · B/I · 1 h
    *Why:* containerise a Python web app, develop, test and add CI. Maps directly onto the Flask service.

92. **Docker: Building best practices** (S) ★
    https://docs.docker.com/build/building/best-practices/
    Docs · I · 30 min
    *Why:* small base images, `.dockerignore`, layer ordering, non-root user, pinning. Use it as the checklist for grading Dockerfiles.

93. **Docker: Multi-stage builds** ★
    https://docs.docker.com/build/building/multi-stage/
    Docs · I · 20 min
    *Why:* build deps in one stage, ship a slim runtime. Critical for keeping ML images small.

94. **Docker: Build cache**
    https://docs.docker.com/build/cache/
    Docs · I · 20 min
    *Why:* why `COPY requirements.txt` then `pip install` goes *before* `COPY . .`. Saves many minutes per rebuild.

95. **Docker: Build context**
    https://docs.docker.com/build/concepts/context/
    Docs · I · 10 min
    *Why:* explains slow builds caused by sending datasets or `.venv` into the context (use `.dockerignore`).

96. **Dockerfile reference**
    https://docs.docker.com/reference/dockerfile/
    Docs · I · reference
    *Why:* exact semantics of `CMD` vs `ENTRYPOINT`, `EXPOSE`, `HEALTHCHECK`, `ARG` / `ENV`.

97. **Docker Compose: overview** (S)
    https://docs.docker.com/compose/
    Docs · B · 10 min
    *Why:* what Compose is for: running Flask + Streamlit together locally.

98. **Docker Compose: Quickstart** (S) ★
    https://docs.docker.com/compose/gettingstarted/
    Tut · B/I · 30 min
    *Why:* services, networks (Streamlit calls `http://api:5000`), ports, and `depends_on`. Template for the two-service app.

99. **Docker: GitHub Actions for Docker builds**
    https://docs.docker.com/build/ci/github-actions/
    Docs · I · 30 min
    *Why:* build and push images in CI, including multi-platform (arm64 for Oracle A1). A bridge to MLOps.

---

## 9. Kubernetes (minikube / kind, Deployments, Services, probes, HPA)

100. **Kubernetes: Overview** (S)
     https://kubernetes.io/docs/concepts/overview/
     Docs · B · 15 min
     *Why:* why orchestration exists, and the declarative desired-state model.

101. **Learn Kubernetes Basics (official tutorial)** (S) ★
     https://kubernetes.io/docs/tutorials/kubernetes-basics/
     Tut · B · 1.5 h
     *Why:* create cluster → deploy → explore → expose → scale → update. The ideal skeleton for the K8s day.

102. **minikube start** (S) ★
     https://minikube.sigs.k8s.io/docs/start/
     Docs · B · 20 min
     *Why:* installation per OS and the first deployment. Have students do this **before** class.

103. **minikube: Pushing images** (S)
     https://minikube.sigs.k8s.io/docs/handbook/pushing/
     Docs · I · 15 min
     *Why:* `minikube image load` / `docker-env`. Fixes the classic `ErrImagePull` for locally built images.

104. **minikube: Accessing apps**
     https://minikube.sigs.k8s.io/docs/handbook/accessing/
     Docs · B · 15 min
     *Why:* NodePort, `minikube service` and `minikube tunnel` for LoadBalancer. How students open the app in a browser.

105. **kind: Quick Start** (S) ★
     https://kind.sigs.k8s.io/docs/user/quick-start/
     Docs · B/I · 30 min
     *Why:* K8s-in-Docker (lightweight, multi-node configs, CI-friendly). Includes `kind load docker-image` for local images.

106. **Deployments** ★
     https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
     Docs · I · 40 min
     *Why:* ReplicaSets, rolling updates, rollback. Demo updating the model image version live.

107. **Service** ★
     https://kubernetes.io/docs/concepts/services-networking/service/
     Docs · I · 40 min
     *Why:* ClusterIP / NodePort / LoadBalancer, selectors, DNS. How Streamlit finds the Flask API inside the cluster.

108. **Use Port Forwarding to Access Applications in a Cluster** (S)
     https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/
     Docs · B · 10 min
     *Why:* the quickest way to reach a pod or service while debugging.

109. **Pod Lifecycle**
     https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/
     Docs · I · 30 min
     *Why:* phases, container states and restart policy. Needed to interpret `CrashLoopBackOff`.

110. **Configure Liveness, Readiness and Startup Probes** (S) ★
     https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
     Docs · I · 30 min
     *Why:* ML containers load models slowly. Use a **startupProbe** and a readiness probe on `/healthz` so traffic arrives only once the model is loaded.

111. **Resource Management for Pods and Containers** ★
     https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
     Docs · I · 30 min
     *Why:* requests vs limits and OOMKilled. HPA CPU% is computed relative to **requests**, so they must be set.

112. **Horizontal Pod Autoscaling (concept)**
     https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/
     Docs · I/A · 30 min
     *Why:* the scaling algorithm, stabilisation windows and metrics types.

113. **HorizontalPodAutoscaler Walkthrough** (S) ★
     https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/
     Tut · I · 45 min
     *Why:* a step-by-step load-generator demo. Reuse it as the in-class HPA exercise.

114. **metrics-server (GitHub)**
     https://github.com/kubernetes-sigs/metrics-server
     Tool/Docs · I · 10 min
     *Why:* HPA needs it. On minikube use `minikube addons enable metrics-server`; on kind you need the `--kubelet-insecure-tls` arg.

115. **ConfigMaps**
     https://kubernetes.io/docs/concepts/configuration/configmap/
     Docs · B/I · 15 min
     *Why:* inject `API_URL` and model version without rebuilding images.

116. **Secrets**
     https://kubernetes.io/docs/concepts/configuration/secret/
     Docs · I · 20 min
     *Why:* handling tokens (e.g. an HF token). Note that Secrets are only base64-encoded, not encrypted by default.

117. **Ingress**
     https://kubernetes.io/docs/concepts/services-networking/ingress/
     Docs · I · 20 min
     *Why:* path-based routing (`/api` → Flask, `/` → Streamlit) on a single host. An optional extension.

118. **kubectl Quick Reference** (S)
     https://kubernetes.io/docs/reference/kubectl/quick-reference/
     Docs · B · reference
     *Why:* print it as the student cheat-sheet.

---

## 10. Deploying on free cloud tiers (details in `free_cloud_options.md`)

119. **Hugging Face: Docker Spaces** ★
     https://huggingface.co/docs/hub/spaces-sdks-docker
     Docs · I · 20 min
     *Why:* `sdk: docker`, `app_port` (default 7860), UID 1000 permissions, secrets. **Note: creating Docker Spaces now requires HF PRO.**

120. **Hugging Face: Your First Docker Space**
     https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo
     Tut · B/I · 30 min
     *Why:* an end-to-end FastAPI example you can adapt to Flask.

121. **Hugging Face: Spaces Overview (hardware, sleep, plan requirements)**
     https://huggingface.co/docs/hub/spaces-overview
     Docs · B · 15 min
     *Why:* free CPU Basic specs, the paid-plan requirement, and open network ports (80/443/8080).

122. **Render: Deploy for Free** (S) ★
     https://render.com/docs/free
     Docs · B · 15 min
     *Why:* spin-down after 15 min, 750 h/month, ephemeral filesystem. Set student expectations about cold starts.

123. **Render: Docker on Render** (S)
     https://render.com/docs/docker
     Docs · B/I · 20 min
     *Why:* deploy from a Dockerfile or a prebuilt image. The primary no-card Docker demo path.

124. **Render: Deploy a Flask App**
     https://render.com/docs/deploy-flask
     Tut · B · 15 min
     *Why:* the Gunicorn start command and env vars on Render.

125. **Oracle Cloud: Always Free Resources**
     https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm
     Docs · I · 20 min
     *Why:* current A1 limits (2 OCPU / 12 GB as of June 2026) and the idle-reclamation rules.

126. **Oracle OKE: Comparing Enhanced vs Basic Clusters**
     https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengcomparingenhancedwithbasicclusters_topic.htm
     Docs · I · 15 min
     *Why:* basic clusters have no control-plane fee. This is how to run managed K8s at $0.

127. **Google Cloud: Free features and trial offer**
     https://docs.cloud.google.com/free/docs/free-cloud-features
     Docs · B · 20 min
     *Why:* $300 / 90-day trial, e2-micro regions, Cloud Run and GKE free-tier details.

128. **AWS: Free Tier (credit-based, since July 2025)**
     https://aws.amazon.com/free/
     Docs · B · 10 min
     *Why:* $100 + $100 credits, 6-month Free plan. Many online tutorials still describe the old 12-month model.

129. **Azure for Students** (S)
     https://azure.microsoft.com/en-us/free/students/
     Docs · B · 5 min
     *Why:* $100, no credit card, school email. The recommended student cloud account.

130. **AKS Free, Standard and Premium tiers**
     https://learn.microsoft.com/en-us/azure/aks/free-standard-pricing-tiers
     Docs · I · 15 min
     *Why:* `--tier free` gives free cluster management; you only pay for nodes.

131. **Killercoda: Kubernetes playground** (S)
     https://killercoda.com/playgrounds/scenario/kubernetes
     Tool · B · n/a
     *Why:* a zero-install real cluster in the browser (1 h sessions). Plan B for students whose laptops can't run minikube.

132. **Google Colab FAQ** (S)
     https://research.google.com/colaboratory/faq.html
     Docs · B · 10 min
     *Why:* realistic expectations (max 12 h, GPUs not guaranteed). Where students train before deploying.

---

## 11. MLOps basics

133. **Sculley et al. (2015): "Hidden Technical Debt in Machine Learning Systems"** (S) ★
     https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
     Paper · I · 45 min
     *Why:* "ML code is a small box in a big system." The founding argument for MLOps. A must-discuss paper.

134. **Google: Rules of Machine Learning (Zinkevich)** (S) ★
     https://developers.google.com/machine-learning/guides/rules-of-ml
     Guide · I · 1 h
     *Why:* 43 battle-tested rules (start simple, get the pipeline right first, monitor). Excellent seminar material.

135. **Google Cloud: MLOps: Continuous delivery and automation pipelines in ML** ★
     https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
     Guide · I · 45 min
     *Why:* the MLOps maturity levels 0/1/2. A clear framework for the final session.

136. **ml-ops.org** (S)
     https://ml-ops.org/
     Guide · B/I · 1 h (browse)
     *Why:* vendor-neutral principles (versioning, testing, monitoring, reproducibility) with diagrams.

137. **Made With ML (Goku Mohandas)** (S) ★
     https://madewithml.com/
     Course · I · many hours
     *Why:* the best free end-to-end "ML in production" course (design, data, testing, CI/CD, serving, monitoring).

138. **Made With ML: GitHub repo**
     https://github.com/GokuMohandas/Made-With-ML
     Code · I/A · reference
     *Why:* production-style code structure to show students what "beyond the notebook" looks like.

139. **Full Stack Deep Learning 2022 (course)** (S)
     https://fullstackdeeplearning.com/course/2022/
     Course · I · ~15 h video
     *Why:* lectures on deployment, monitoring, data management and project structure from practitioners.

140. **Chip Huyen: Machine Learning Systems Design (free notes)** (S)
     https://huyenchip.com/machine-learning-systems-design/toc.html
     Book/notes · I · 3 h
     *Why:* the system-design lens (requirements, data, serving, monitoring) behind her O'Reilly book.

141. **Chip Huyen: "Designing ML Systems" book resources**
     https://github.com/chiphuyen/dmls-book
     Notes · I · 1 h
     *Why:* chapter summaries and curated resources. A good way to cover the book's ideas without assigning the whole book.

142. **Chip Huyen: "What I learned from looking at 200 machine learning tools"**
     https://huyenchip.com/2020/06/22/mlops.html
     Blog · I · 30 min
     *Why:* a map of the MLOps tooling landscape. Helps answer "which tool should we use?"

143. **Chip Huyen: "Real-time machine learning: challenges and solutions"**
     https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html
     Blog · A · 40 min
     *Why:* online prediction vs batch, and streaming features. Good advanced reading.

144. **MLflow documentation** (S)
     https://mlflow.org/docs/latest/index.html
     Docs · B/I · 1 h
     *Why:* experiment tracking and model registry. The simplest MLOps tool to demo on top of the training scripts.

145. **DVC: Get Started** (S)
     https://doc.dvc.org/start
     Docs · B/I · 45 min
     *Why:* data and model versioning alongside Git. Pairs naturally with GitHub-based deploys.

146. **DataTalksClub: MLOps Zoomcamp (free course)** (S)
     https://github.com/DataTalksClub/mlops-zoomcamp
     Course · I · many hours
     *Why:* free, hands-on (tracking, orchestration, deployment, monitoring). Point motivated students here after the course.

147. **Evidently AI: "What is data drift in ML?"** (S)
     https://www.evidentlyai.com/ml-in-production/data-drift
     Blog · B/I · 20 min
     *Why:* clear explanation of data vs concept drift and how to detect it. The "monitoring" piece of MLOps.

148. **GitHub Docs: Publishing Docker images (Actions)**
     https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images
     Docs · I · 30 min
     *Why:* CI that builds and pushes the app image on each commit. The minimal viable CI/CD demo.

149. **KServe**
     https://kserve.github.io/website/
     Docs · A · 30 min
     *Why:* how model serving is done "properly" on Kubernetes (autoscaling, canary). A "what's next" pointer.

150. **eugeneyan/applied-ml (curated industry papers)**
     https://github.com/eugeneyan/applied-ml
     List · I/A · browse
     *Why:* real company write-ups on ML in production. A great pool for MSc case-study assignments.

151. **Jeremy Jordan: "Organizing machine learning projects"** (S)
     https://www.jeremyjordan.me/ml-projects-guide/
     Blog · B/I · 30 min
     *Why:* project scoping, baselines and iteration. Useful framing for student capstones.