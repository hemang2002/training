"""
Tests for the Flask API.

Run from the Day3 folder:   pytest -v tests

Two groups:
  * tests using a TINY fake ONNX model built on the fly -> always runnable, fast, no Day-2 artifacts needed
  * tests using the REAL models in <repo>/models        -> skipped if Day 2 has not produced them yet
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

API_DIR = Path(__file__).resolve().parents[1] / "app" / "api"
sys.path.insert(0, str(API_DIR))

from model_service import ModelService, find_model_dir  # noqa: E402

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def png_bytes(color=(255, 0, 0), size=(32, 32), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def fake_model_dir(tmp_path_factory):
    """Build a 1-layer ONNX 'classifier': GlobalAveragePool -> Flatten -> MatMul(3x10)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(size=(3, 10)).astype(np.float32), name="W")
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
         helper.make_node("Flatten", ["gap"], ["flat"]),
         helper.make_node("MatMul", ["flat", "W"], ["logits"])],
        "fake", [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 96, 96])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 10])], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    d = tmp_path_factory.mktemp("models")
    onnx.save(model, d / "model_fp32.onnx")
    onnx.save(model, d / "model_int8.onnx")
    card = {"img_size": 96, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "labels": LABELS,
            "variants": {"fp32": {"file": "model_fp32.onnx", "accuracy": 0.9},
                         "int8": {"file": "model_int8.onnx", "accuracy": 0.89}}}
    (d / "model_card.json").write_text(json.dumps(card))
    return d


@pytest.fixture()
def client(fake_model_dir):
    from app import create_app
    app = create_app(ModelService(fake_model_dir))
    app.testing = True
    return app.test_client()


# ------------------------------------------------------------------ fake-model tests
def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert set(r.get_json()["models"]) == {"fp32", "int8"}


def test_models(client):
    data = client.get("/models").get_json()
    assert data["labels"] == LABELS
    assert data["variants"]["int8"]["default"] is True


def test_predict_multipart(client):
    r = client.post("/predict?model=fp32&top_k=5", data={"file": (io.BytesIO(png_bytes()), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["prediction"] in LABELS
    assert len(body["top_k"]) == 5
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3


def test_predict_base64(client):
    import base64
    b64 = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    r = client.post("/predict", json={"image_b64": b64})
    assert r.status_code == 200
    assert r.get_json()["model"] == "int8"


def test_grayscale_and_rgba_accepted(client):
    for mode, color in [("L", 128), ("RGBA", (0, 0, 255, 128))]:
        r = client.post("/predict", data={"file": (io.BytesIO(png_bytes(color, mode=mode)), "x.png")},
                        content_type="multipart/form-data")
        assert r.status_code == 200


def test_compare(client):
    r = client.post("/compare", data={"file": (io.BytesIO(png_bytes()), "x.png")},
                    content_type="multipart/form-data")
    body = r.get_json()
    assert set(body["results"]) == {"fp32", "int8"}
    assert body["agree"] is True  # identical fake models must agree


@pytest.mark.parametrize("kwargs,status", [
    ({}, 400),                                                                  # no image at all
    ({"data": {"file": (io.BytesIO(b"not an image"), "x.png")}}, 400),          # garbage bytes
    ({"json": {"image_b64": "%%%notbase64"}}, 400),                             # bad base64
])
def test_bad_requests(client, kwargs, status):
    assert client.post("/predict", **kwargs).status_code == status


def test_unknown_model(client):
    r = client.post("/predict?model=fp64", data={"file": (io.BytesIO(png_bytes()), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 404


def test_metrics_counts(client):
    client.post("/predict", data={"file": (io.BytesIO(png_bytes()), "x.png")}, content_type="multipart/form-data")
    m = client.get("/metrics").get_json()
    assert m["requests"] >= 1


# ------------------------------------------------------------------ real-model tests
REAL_DIR = find_model_dir()
needs_real = pytest.mark.skipif(not (REAL_DIR / "model_fp32.onnx").exists(),
                                reason="Day-2 models not generated yet")


@needs_real
def test_real_models_on_samples():
    """fp32 and int8 should both classify the bundled CIFAR sample images mostly correctly."""
    svc = ModelService(REAL_DIR)
    samples = sorted((Path(__file__).resolve().parents[1] / "app" / "ui" / "samples").glob("*.png"))
    if not samples:
        pytest.skip("no sample images")
    for variant in svc.sessions:
        correct = sum(svc.predict(p.read_bytes(), variant)["prediction"] == p.stem.split("_")[0] for p in samples)
        assert correct / len(samples) >= 0.6, f"{variant}: only {correct}/{len(samples)} correct"
