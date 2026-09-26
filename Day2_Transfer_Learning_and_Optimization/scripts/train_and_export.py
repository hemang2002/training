"""
Day 2 - one-command pipeline: fine-tune -> export -> quantize -> evaluate -> model card.

    python scripts/train_and_export.py                 # default settings (~10k train imgs, 3 epochs)
    python scripts/train_and_export.py --epochs 1 --train-subset 2000 --test-subset 500   # smoke test

What it does (the whole of Day 2 in one file, so the trainer can always produce
the Day-3 serving artefacts with a single command):

  1. Fine-tune an ImageNet-pretrained torchvision MobileNetV2 on a CIFAR-10 subset
     (classifier[1] replaced by nn.Linear(1280, 10), images resized to 96x96).
  2. Save the state_dict            -> models/mobilenetv2_cifar10_fp32.pth
  3. Export FP32 ONNX (opset 17)    -> models/model_fp32.onnx
     input name "input", output name "logits", dynamic batch axis, raw logits.
  4. Static INT8 quantization with ONNX Runtime (QDQ, per-channel, calibrated
     on real training images)       -> models/model_int8.onnx
  5. Evaluate both ONNX models (accuracy on a fixed test subset, size, latency).
  6. Write models/labels.json and models/model_card.json (the Day-3 API contract).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent  # .../Trtaining

LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]
MEAN = [0.485, 0.456, 0.406]   # ImageNet statistics - the backbone was pretrained with these
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=img_size // 12, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # EXACTLY the Day-3 inference preprocessing (contract)
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, test_tf


def fixed_subset(n_total: int, n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_total, generator=g)[:n].tolist()


def build_model(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(LABELS))  # 1280 -> 10, keep Dropout
    return model


@torch.no_grad()
def evaluate_torch(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def file_mb(path: Path) -> float:
    return os.path.getsize(path) / 1e6


# --------------------------------------------------------------------------- training
def train(model, train_loader, test_loader, epochs: int, lr: float):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.15)
    for epoch in range(epochs):
        model.train()
        t0, running, seen, correct = time.time(), 0.0, 0, 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            if (i + 1) % 25 == 0:
                print(f"  epoch {epoch+1} step {i+1}/{len(train_loader)} "
                      f"loss {running/seen:.3f} train-acc {correct/seen:.3f}", flush=True)
        test_acc = evaluate_torch(model, test_loader)
        print(f"Epoch {epoch+1}/{epochs}: loss {running/seen:.3f} train-acc {correct/seen:.3f} "
              f"test-acc {test_acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


# --------------------------------------------------------------------------- ONNX
def export_onnx(model: nn.Module, path: Path, img_size: int) -> None:
    import warnings
    import onnx
    # the legacy (TorchScript) exporter is chosen deliberately -> silence its deprecation notice
    warnings.filterwarnings("ignore", message=r".*legacy TorchScript-based ONNX export.*")
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,          # legacy TorchScript exporter: stable dynamic_axes support
    )
    m = onnx.load(str(path))   # loads external data too, if any was written
    onnx.checker.check_model(m)
    onnx.save(m, str(path), save_as_external_data=False)  # guarantee a single self-contained file
    stale = Path(str(path) + ".data")
    if stale.exists():
        stale.unlink()


def quantize_onnx_static(fp32_path: Path, int8_path: Path, calib_batches: list[np.ndarray]) -> None:
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, batches):
            self._it = iter([{"input": b} for b in batches])

        def get_next(self):
            return next(self._it, None)

    prep_path = int8_path.with_name(int8_path.stem + "_prep.onnx")
    # symbolic shape inference + ONNX optimisations (e.g. folds constants) before quantization
    quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=False)
    try:
        quantize_static(
            model_input=str(prep_path),
            model_output=str(int8_path),
            calibration_data_reader=Reader(calib_batches),
            quant_format=QuantFormat.QDQ,           # QuantizeLinear/DequantizeLinear pairs
            per_channel=True,                       # one scale per output channel for weights
            activation_type=QuantType.QUInt8,       # U8S8: recommended on x86 CPUs
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
    finally:
        if prep_path.exists():
            prep_path.unlink()
    m = onnx.load(str(int8_path))
    onnx.checker.check_model(m)
    onnx.save(m, str(int8_path), save_as_external_data=False)


def evaluate_onnx(path: Path, xs: np.ndarray, ys: np.ndarray, batch: int = 100, threads: int = 4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads   # fixed thread count -> comparable, reproducible latency
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    correct = 0
    for i in range(0, len(xs), batch):
        logits = sess.run(["logits"], {"input": xs[i:i + batch]})[0]
        correct += int((logits.argmax(1) == ys[i:i + batch]).sum())
    acc = correct / len(xs)
    # latency, batch = 1, after warm-up (median is robust to OS / other-process noise)
    x1 = xs[:1]
    for _ in range(20):
        sess.run(None, {"input": x1})
    times = []
    for _ in range(200):
        t = time.perf_counter()
        sess.run(None, {"input": x1})
        times.append((time.perf_counter() - t) * 1000)
    return acc, float(np.mean(times)), float(np.median(times))


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="Day 2: fine-tune MobileNetV2 on CIFAR-10 and export FP32/INT8 ONNX")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-subset", type=int, default=10000)
    p.add_argument("--test-subset", type=int, default=2000)
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--calib-size", type=int, default=320, help="training images used for INT8 calibration")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bench-threads", type=int, default=4, help="ONNX Runtime intra-op threads for latency")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "models")
    args = p.parse_args()

    t_start = time.time()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | out -> {args.out}")

    train_tf, test_tf = build_transforms(args.img_size)
    train_full = datasets.CIFAR10(str(args.data), train=True, download=True, transform=train_tf)
    test_full = datasets.CIFAR10(str(args.data), train=False, download=True, transform=test_tf)
    calib_full = datasets.CIFAR10(str(args.data), train=True, download=False, transform=test_tf)
    assert train_full.classes == LABELS, train_full.classes

    train_idx = fixed_subset(len(train_full), args.train_subset, args.seed)
    test_idx = fixed_subset(len(test_full), args.test_subset, args.seed)
    train_ds, test_ds = Subset(train_full, train_idx), Subset(test_full, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False, num_workers=args.workers)
    print(f"train images: {len(train_ds)} | test images: {len(test_ds)} | img_size {args.img_size}")

    # 1) fine-tune -------------------------------------------------------------------
    model = build_model(pretrained=True)
    t0 = time.time()
    train(model, train_loader, test_loader, args.epochs, args.lr)
    train_time = time.time() - t0
    torch_acc = evaluate_torch(model, test_loader)
    print(f"PyTorch FP32 test accuracy: {torch_acc:.4f}  (training {train_time/60:.1f} min)")

    # 2) save state_dict ---------------------------------------------------------------
    pth = args.out / "mobilenetv2_cifar10_fp32.pth"
    torch.save(model.state_dict(), pth)

    # 3) export FP32 ONNX -------------------------------------------------------------
    fp32_onnx = args.out / "model_fp32.onnx"
    export_onnx(model, fp32_onnx, args.img_size)

    # test set as a numpy array (contract preprocessing)
    xs = np.stack([test_ds[i][0].numpy() for i in range(len(test_ds))]).astype(np.float32)
    ys = np.array([test_ds[i][1] for i in range(len(test_ds))])

    import onnxruntime as ort
    sess = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(torch.from_numpy(xs[:8])).numpy()
    got = sess.run(["logits"], {"input": xs[:8]})[0]
    max_diff = float(np.abs(ref - got).max())
    print(f"torch vs onnxruntime max |diff| on 8 images: {max_diff:.2e}")
    assert np.allclose(ref, got, atol=1e-3), "ONNX export does not match PyTorch!"

    # 4) static INT8 quantization (calibration on training images, test-time preprocessing)
    calib_idx = fixed_subset(len(calib_full), args.calib_size, args.seed + 1)
    calib = np.stack([calib_full[i][0].numpy() for i in calib_idx]).astype(np.float32)
    calib_batches = [calib[i:i + 32] for i in range(0, len(calib), 32)]
    int8_onnx = args.out / "model_int8.onnx"
    quantize_onnx_static(fp32_onnx, int8_onnx, calib_batches)

    # 5) evaluate ---------------------------------------------------------------------
    variants = {}
    for name, path in [("fp32", fp32_onnx), ("int8", int8_onnx)]:
        acc, lat_mean, lat_med = evaluate_onnx(path, xs, ys, threads=args.bench_threads)
        variants[name] = {"file": path.name, "size_mb": round(file_mb(path), 3),
                          "accuracy": round(acc, 4), "latency_ms": round(lat_med, 3)}
        print(f"{name}: size {file_mb(path):.2f} MB | acc {acc:.4f} | latency mean {lat_mean:.2f} ms "
              f"(median {lat_med:.2f} ms, batch=1, CPU)")

    # 6) labels + model card ----------------------------------------------------------
    (args.out / "labels.json").write_text(json.dumps(LABELS, indent=2))
    card = {
        "img_size": args.img_size,
        "mean": MEAN,
        "std": STD,
        "labels": LABELS,
        "input_name": "input",
        "output_name": "logits",
        "variants": variants,
        "trained_on": "CIFAR-10 subset",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "details": {
            "architecture": "torchvision mobilenet_v2 (ImageNet-pretrained), classifier[1]=Linear(1280,10)",
            "train_images": len(train_ds), "test_images": len(test_ds), "epochs": args.epochs,
            "calibration_images": len(calib), "pytorch_fp32_accuracy": round(torch_acc, 4),
            "onnx_opset": 17,
            "latency_note": f"median of 200 runs, batch=1, ONNX Runtime CPU, {args.bench_threads} threads",
            "int8_scheme": "ONNX Runtime static, QDQ, per-channel, U8S8, MinMax",
            "preprocessing": "RGB -> Resize((96,96)) -> ToTensor -> Normalize(mean,std) -> NCHW float32",
            "output": "raw logits (apply softmax in the client)",
        },
    }
    (args.out / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card["variants"], indent=2))
    print(f"Done in {(time.time()-t_start)/60:.1f} min. Files in {args.out}:")
    for f in sorted(args.out.iterdir()):
        print(f"  {f.name:35s} {file_mb(f):8.2f} MB")


if __name__ == "__main__":
    main()
