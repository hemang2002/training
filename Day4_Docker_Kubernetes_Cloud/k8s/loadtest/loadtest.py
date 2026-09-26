"""
loadtest.py - hammer the API to watch Kubernetes load-balancing and autoscaling (HPA).

Uses only the standard library + Pillow (both are inside the cifar-api image), so it can run:
  * from your laptop :  python loadtest.py --url http://localhost:5000 --duration 60
  * inside the cluster:  kubectl apply -f k8s/loadtest/loadtest-job.yaml   (see README)

It prints requests/s, latency percentiles and WHICH POD answered - proof that the Service spreads traffic.
"""
import argparse
import base64
import io
import json
import threading
import time
import urllib.request
from collections import Counter

from PIL import Image


def make_payload() -> bytes:
    img = Image.effect_noise((96, 96), 64).convert("RGB")   # random image, no files needed
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return json.dumps({"image_b64": base64.b64encode(buf.getvalue()).decode()}).encode()


def worker(url, payload, deadline, lat, pods, errors, lock):
    req_url = f"{url}/predict?model=fp32"   # fp32 = more CPU per request -> scales sooner
    while time.time() < deadline:
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(req_url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                pod = json.loads(r.read())["pod"]
            with lock:
                lat.append((time.perf_counter() - t0) * 1000)
                pods[pod] += 1
        except Exception:  # noqa: BLE001
            with lock:
                errors[0] += 1
            time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://cifar-api:5000")
    ap.add_argument("--duration", type=int, default=120, help="seconds")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--report-every", type=int, default=10)
    a = ap.parse_args()

    payload = make_payload()
    lat, pods, errors, lock = [], Counter(), [0], threading.Lock()
    deadline = time.time() + a.duration
    threads = [threading.Thread(target=worker, args=(a.url.rstrip("/"), payload, deadline, lat, pods, errors, lock),
                                daemon=True) for _ in range(a.concurrency)]
    print(f"Load test: {a.url}  concurrency={a.concurrency}  duration={a.duration}s", flush=True)
    for t in threads:
        t.start()

    start, last_n = time.time(), 0
    while any(t.is_alive() for t in threads):
        time.sleep(a.report_every)
        with lock:
            n, snap, p = len(lat), sorted(lat[last_n:]), dict(pods)
        if snap:
            p50, p95 = snap[len(snap) // 2], snap[int(0.95 * (len(snap) - 1))]
            print(f"[{time.time() - start:5.0f}s] {(n - last_n) / a.report_every:6.1f} req/s  "
                  f"p50={p50:6.1f}ms  p95={p95:6.1f}ms  errors={errors[0]}  pods={len(p)}", flush=True)
        last_n = n

    print("\nRequests per pod (load balancing):")
    for pod, c in pods.most_common():
        print(f"  {pod:45s} {c}")
    print(f"total={len(lat)} errors={errors[0]}")


if __name__ == "__main__":
    main()
