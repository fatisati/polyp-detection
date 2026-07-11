"""
Isolates where modal_ms (see latency_bench.py) actually goes: RPC/invocation
overhead vs. the GPU forward pass itself.

Calls PolypDetector.infer_frame_bench directly through the Modal SDK — same
call pattern as backend/services/modal_client.py (`.remote.aio()`, one in
flight at a time) but bypassing FastAPI/WebSocket entirely, so this measures
the backend-server <-> Modal leg in isolation. infer_frame_bench returns
decode_ms and gpu_ms measured *inside* the container; rpc_ms is timed here.

    overhead_ms = rpc_ms - decode_ms - gpu_ms

is everything that isn't decode or compute: Modal's invocation/scheduling,
serialization, and the network hop between this machine and Modal's region.

Requires MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment (source
backend/.env before running — same credentials the backend uses).

Usage:
    set -a && source backend/.env && set +a
    python experiments/modal_rpc_bench.py --widths 160,320,640 --frames 30
"""

import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path

import cv2
import modal
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


def resize_and_encode(frame: np.ndarray, width: int, quality: float = 0.85) -> bytes:
    h, w = frame.shape[:2]
    scale = width / w
    resized = cv2.resize(frame, (width, round(h * scale)))
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, int(quality * 100)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


async def bench_width(detector, frame: np.ndarray, width: int, n_frames: int, n_warmup: int) -> list[dict]:
    jpeg = resize_and_encode(frame, width)
    results = []
    for i in range(n_warmup + n_frames):
        t0 = time.perf_counter()
        r = await detector.infer_frame_bench.remote.aio(jpeg)
        rpc_ms = (time.perf_counter() - t0) * 1000

        if i >= n_warmup:
            overhead_ms = rpc_ms - r["decode_ms"] - r["gpu_ms"]
            results.append({
                "width": width, "frame_idx": i,
                "rpc_ms": round(rpc_ms, 2), "decode_ms": r["decode_ms"],
                "gpu_ms": r["gpu_ms"], "overhead_ms": round(overhead_ms, 2),
                "num_boxes": len(r["boxes"]),
            })
        if (i + 1) % 10 == 0:
            print(f"    [{width}px] {i + 1}/{n_warmup + n_frames}")
    return results


async def main_async(args: argparse.Namespace) -> None:
    widths = [int(w) for w in args.widths.split(",")]

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)  # a mid-video frame, not blank
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.video}")

    print("Connecting to PolypDetector...")
    detector = modal.Cls.from_name("polyp-detection", "PolypDetector")()

    all_results = []
    for width in widths:
        print(f"\nBenchmarking width={width} ({args.frames} timed calls, {args.warmup} warmup)")
        res = await bench_width(detector, frame, width, args.frames, args.warmup)
        all_results.extend(res)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"modal_rpc_bench_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\nRaw results: {csv_path}")

    print("\n" + "=" * 78)
    print(f"{'width':>6} {'rpc_mean':>9} {'gpu_mean':>9} {'decode_mean':>12} {'overhead_mean':>14} {'overhead_%':>11}")
    print("=" * 78)
    by_width: dict[int, list[dict]] = {}
    for r in all_results:
        by_width.setdefault(r["width"], []).append(r)
    summary_rows = []
    for width, rs in sorted(by_width.items()):
        rpc = statistics.mean(r["rpc_ms"] for r in rs)
        gpu = statistics.mean(r["gpu_ms"] for r in rs)
        decode = statistics.mean(r["decode_ms"] for r in rs)
        overhead = statistics.mean(r["overhead_ms"] for r in rs)
        pct = overhead / rpc * 100
        print(f"{width:>6} {rpc:>9.1f} {gpu:>9.1f} {decode:>12.1f} {overhead:>14.1f} {pct:>10.1f}%")
        summary_rows.append({"width": width, "rpc_mean_ms": round(rpc, 1), "gpu_mean_ms": round(gpu, 1),
                              "decode_mean_ms": round(decode, 1), "overhead_mean_ms": round(overhead, 1),
                              "overhead_pct": round(pct, 1)})
    print("=" * 78)

    summary_path = RESULTS_DIR / f"modal_rpc_bench_{ts}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary: {summary_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="frontend/public/demos/test_polyp_seq2.mp4")
    p.add_argument("--widths", default="160,320,640")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
