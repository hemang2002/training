"""
prepare_demo.py  --  run this THE NIGHT BEFORE the live demos (needs internet).

It downloads / caches everything the Live_Demos notebooks need, so that the
demo itself works even on bad conference Wi-Fi:

  1. Penn-Fudan pedestrian dataset  ->  <ROOT>/data/PennFudanPed/
  2. FashionMNIST                   ->  <ROOT>/data/FashionMNIST/
  3. COCO-pretrained torchvision detection weights -> torch hub cache
     (~/.cache/torch/hub/checkpoints on Linux/macOS,
      C:\\Users\\<you>\\.cache\\torch\\hub\\checkpoints on Windows)

Usage (from the Live_Demos folder):
    python prepare_demo.py

It is idempotent: anything already present is skipped.
"""
from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = (HERE / ".." / "data").resolve()

# Primary URL (University of Pennsylvania). Verified working 2026-09.
PENNFUDAN_URLS = [
    "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip",
    # Fallback mirror: Internet Archive raw copy of the same file ("id_" = raw
    # bytes, no Wayback HTML wrapper). Verified: identical 53.7 MB zip.
    "https://web.archive.org/web/2023id_/https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip",
]


def _ssl_context():
    """The upenn.edu server does not send its intermediate certificate, so on
    some machines (e.g. Windows + python.org Python) the default SSL context
    fails with CERTIFICATE_VERIFY_FAILED. The `certifi` CA bundle fixes it."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download(url: str, dst: Path) -> None:
    print(f"  downloading {url}")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:6.1f} / {total/1e6:.1f} MB", end="")
    print(f"\n  done in {time.time()-t0:.1f}s")


def prepare_pennfudan() -> None:
    print("[1/3] Penn-Fudan pedestrian dataset")
    target = DATA / "PennFudanPed"
    if (target / "PNGImages").exists() and len(list((target / "PNGImages").glob("*.png"))) == 170:
        print(f"  already present at {target}")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    zpath = DATA / "PennFudanPed.zip"
    last_err = None
    for url in PENNFUDAN_URLS:
        try:
            _download(url, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(DATA)
            break
        except Exception as e:  # try next mirror
            last_err = e
            print(f"  failed: {e!r}")
    else:
        raise RuntimeError(f"Could not download Penn-Fudan: {last_err!r}")
    zpath.unlink(missing_ok=True)
    n_img = len(list((target / "PNGImages").glob("*.png")))
    n_msk = len(list((target / "PedMasks").glob("*.png")))
    print(f"  OK: {n_img} images, {n_msk} masks in {target}")
    assert n_img == 170 and n_msk == 170, "unexpected Penn-Fudan content"


def prepare_fashionmnist() -> None:
    print("[2/3] FashionMNIST")
    from torchvision import datasets
    for attempt in range(1, 6):          # flaky Wi-Fi: retry a few times
        try:
            tr = datasets.FashionMNIST(str(DATA), train=True, download=True)
            te = datasets.FashionMNIST(str(DATA), train=False, download=True)
            break
        except Exception as e:
            print(f"  attempt {attempt} failed: {e!r}; retrying in 5 s")
            time.sleep(5)
    else:
        raise RuntimeError("FashionMNIST download failed 5 times")
    print(f"  OK: train={len(tr)}  test={len(te)}  in {DATA / 'FashionMNIST'}")


def prepare_weights() -> None:
    print("[3/3] COCO-pretrained detection weights (torch hub cache)")
    from torchvision.models import detection as D
    for name, w in [
        ("fasterrcnn_mobilenet_v3_large_fpn", D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
        ("fasterrcnn_mobilenet_v3_large_320_fpn", D.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT),
    ]:
        getattr(D, name)(weights=w)   # triggers download into the hub cache
        print(f"  OK: {name}")
    import torch
    print(f"  cache dir: {Path(torch.hub.get_dir()) / 'checkpoints'}")


if __name__ == "__main__":
    print(f"Data folder: {DATA}")
    prepare_pennfudan()
    prepare_fashionmnist()
    prepare_weights()
    print("\nAll demo assets are ready.")
    sys.exit(0)
