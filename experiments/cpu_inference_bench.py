"""
Local CPU inference benchmark — no network hop, no Modal.

Loads the same weights the Modal app uses (goktug14/yolov5_kvasir_polyp) and
runs the real model directly on this machine's CPU, to compare against the
Modal GPU + transatlantic RPC round trip measured in modal_rpc_bench.py.

If this server's CPU inference time (plus ~0 network overhead, since it's
local) beats the current ~250-260ms Modal round trip, moving inference
onto the backend server itself is a bigger win than anything achievable by
resizing frames or optimizing the GPU call.

Usage (inside experiments/cpu_venv):
    cpu_venv/bin/python experiments/cpu_inference_bench.py --widths 160,320,640 --frames 20
"""

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

HF_REPO = "goktug14/yolov5_kvasir_polyp"
HF_FILE = "weights/best.pt"
WEIGHTS_CACHE = Path(__file__).parent / "cpu_venv_weights" / "yolo_polyp.pt"


def get_weights() -> str:
    if WEIGHTS_CACHE.exists():
        return str(WEIGHTS_CACHE)
    from huggingface_hub import hf_hub_download
    print(f"Downloading weights from {HF_REPO}...")
    tmp = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    WEIGHTS_CACHE.parent.mkdir(exist_ok=True)
    WEIGHTS_CACHE.write_bytes(Path(tmp).read_bytes())
    return str(WEIGHTS_CACHE)


def resize(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, round(h * scale)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="frontend/public/demos/test_polyp_seq2.mp4")
    p.add_argument("--widths", default="160,320,640")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    from ultralytics import YOLO

    weights = get_weights()
    print(f"Loading model from {weights} ...")
    model = YOLO(weights)
    model.to("cpu")

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.video}")

    widths = [int(w) for w in args.widths.split(",")]
    print("\n" + "=" * 60)
    print(f"{'width':>6} {'mean_ms':>9} {'p50_ms':>8} {'p90_ms':>8} {'max_ms':>8}")
    print("=" * 60)
    for width in widths:
        resized = resize(frame, width)
        times = []
        for i in range(args.warmup + args.frames):
            t0 = time.perf_counter()
            model(resized, conf=0.3, verbose=False)
            ms = (time.perf_counter() - t0) * 1000
            if i >= args.warmup:
                times.append(ms)
        print(f"{width:>6} {statistics.mean(times):>9.1f} {statistics.median(times):>8.1f} "
              f"{sorted(times)[int(len(times) * 0.9)]:>8.1f} {max(times):>8.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
