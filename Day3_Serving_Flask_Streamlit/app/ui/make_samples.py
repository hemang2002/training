"""
make_samples.py - export a few CIFAR-10 *test* images as PNGs for the Streamlit UI and the API tests.

Run once (needs torchvision, which is only a dev dependency - the UI itself does not need it):
    python make_samples.py            # 2 images per class -> samples/<label>_<n>.png
"""
from pathlib import Path

from torchvision.datasets import CIFAR10

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]           # <repo>
OUT = HERE / "samples"
PER_CLASS = 2

ds = CIFAR10(root=ROOT / "data", train=False, download=True)
OUT.mkdir(exist_ok=True)
counts = {c: 0 for c in ds.classes}
for img, label in ds:  # deterministic: first N test images of each class
    name = ds.classes[label]
    if counts[name] < PER_CLASS:
        counts[name] += 1
        img.save(OUT / f"{name}_{counts[name]}.png")
    if all(v >= PER_CLASS for v in counts.values()):
        break
print(f"Saved {sum(counts.values())} images to {OUT}")
