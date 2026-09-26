"""
sync_from_day3.py - copy the latest app code, sample images and models into ml-app/ before building images.

    python sync_from_day3.py

Day 3 is where we develop; Day 4's ml-app/ is the self-contained "deployable unit"
(everything a Docker build context needs, nothing else).
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../Day4_Docker_Kubernetes_Cloud/ml-app
ROOT = HERE.parents[1]                             # repo root
DAY3 = ROOT / "Day3_Serving_Flask_Streamlit" / "app"

copies = [
    (DAY3 / "api" / "app.py", HERE / "api" / "app.py"),
    (DAY3 / "api" / "model_service.py", HERE / "api" / "model_service.py"),
    (DAY3 / "api" / "requirements.txt", HERE / "api" / "requirements.txt"),
    (DAY3 / "ui" / "streamlit_app.py", HERE / "ui" / "streamlit_app.py"),
    (DAY3 / "ui" / "requirements.txt", HERE / "ui" / "requirements.txt"),
]
for src, dst in copies:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src.relative_to(ROOT)}")

shutil.copytree(DAY3 / "ui" / "samples", HERE / "ui" / "samples", dirs_exist_ok=True)
print("copied ui/samples/")

(HERE / "models").mkdir(exist_ok=True)
for name in ["model_fp32.onnx", "model_int8.onnx", "labels.json", "model_card.json"]:
    src = ROOT / "models" / name
    if not src.exists():
        raise SystemExit(f"Missing {src} - run Day2_Transfer_Learning_and_Optimization/scripts/train_and_export.py first")
    shutil.copy2(src, HERE / "models" / name)
    print(f"copied models/{name}  ({src.stat().st_size / 1e6:.1f} MB)")
print("ml-app is ready to build.")
