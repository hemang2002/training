"""
deploy_hf_space.py - publish the all-in-one app (API + UI) to a Hugging Face Space (Docker SDK, free CPU).

    pip install huggingface_hub
    hf auth login                         # or: set HF_TOKEN=<write token>   (older CLI: huggingface-cli login)
    python deploy_hf_space.py <your-hf-username>/cifar10-demo

Why a script instead of `git push`? The Hub stores binary files (our .onnx models, .png samples) through its
large-file storage; `upload_folder` handles that automatically, while a plain git push of binaries is rejected.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent

SPACE_README = """---
title: CIFAR-10 Classifier
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: MobileNetV2 fp32 vs int8 served by Flask + ONNX Runtime
---

# CIFAR-10 Classifier (training demo)

MobileNetV2 fine-tuned on CIFAR-10, exported to ONNX (fp32 + int8), served by a Flask API with
ONNX Runtime and a Streamlit UI - both running in this single container.
"""


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        raise SystemExit("usage: python deploy_hf_space.py <hf-username>/<space-name>")
    repo_id = sys.argv[1]

    for required in ["models/model_fp32.onnx", "models/model_int8.onnx", "models/model_card.json", "ui/samples"]:
        if not (HERE / required).exists():
            raise SystemExit(f"missing {required} - run `python sync_from_day3.py` first")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        # The Space expects a file literally named "Dockerfile" at the repo root.
        shutil.copy2(HERE / "Dockerfile.allinone", stage / "Dockerfile")
        # Write start.sh with LF line endings no matter what OS we are on.
        (stage / "start.sh").write_bytes((HERE / "start.sh").read_bytes().replace(b"\r\n", b"\n"))
        (stage / "README.md").write_text(SPACE_README, encoding="utf-8")
        for sub in ["api", "ui", "models"]:
            shutil.copytree(HERE / sub, stage / sub,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "Dockerfile*", "*.pth"))

        api = HfApi()
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="space",
                          commit_message="Deploy CIFAR-10 demo")

    print(f"Uploaded. Build logs + app: https://huggingface.co/spaces/{repo_id}")
    print("First build takes a few minutes; status goes Building -> Running.")


if __name__ == "__main__":
    main()
