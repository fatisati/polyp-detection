"""
Latency benchmark for the live per-frame inference pipeline.

Hits the real backend over the same protocol the frontend uses (binary JPEG
frames over WebSocket at /api/ws/infer, one frame in flight at a time — no
pipelining, matching RealtimePlayer.tsx), across a matrix of resize
width x JPEG quality configs, and logs latency for every single frame:

  client_e2e_ms  - full round trip as the browser would see it (send -> boxes back).
                   This is the number that determines how laggy the boxes look.
  modal_ms       - time backend spent waiting on the Modal RPC (network to Modal + GPU inference).
  total_ms       - backend's own total processing time for the frame.
  network_ms     - client_e2e_ms - total_ms: WS transport + framing overhead
                   between browser and backend, not visible in server timing.
  payload_bytes  - size of the JPEG actually sent.

Use this to pick a (width, quality) config that hits your target latency
*before* wiring up a live camera — resizing is the main lever we have on
client_e2e_ms since it shrinks both the upload and the GPU forward pass.

Setup:
    pip install -r experiments/requirements.txt

Usage:
    python experiments/latency_bench.py --http-url http://YOUR_SERVER:8000

    # Custom matrix, more frames per config, stricter target
    python experiments/latency_bench.py \
        --http-url http://YOUR_SERVER:8000 \
        --widths 160,224,320,480 \
        --qualities 0.6,0.85 \
        --frames 80 --target-ms 150

Output:
    - Per-frame CSV: experiments/results/latency_<timestamp>.csv
    - Per-config summary CSV: experiments/results/latency_<timestamp>_summary.csv
    - Summary table printed to stdout
"""

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import websockets

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class FrameResult:
    width: int
    quality: float
    repeat: int
    frame_idx: int
    payload_bytes: int
    ok: bool
    client_e2e_ms: float = 0.0
    recv_ms: int = 0
    modal_ms: int = 0
    total_ms: int = 0
    network_ms: float = 0.0
    num_boxes: int = 0
    error: str = ""


def load_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"No frames decoded from: {video_path}")
    return frames


def resize_and_encode(frame: np.ndarray, width: int, quality: float) -> bytes:
    h, w = frame.shape[:2]
    scale = width / w
    resized = cv2.resize(frame, (width, round(h * scale)))
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, int(quality * 100)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def warmup_backend(http_url: str, timeout_s: float = 90.0) -> None:
    print(f"Warming up Modal container via {http_url}/api/session/start ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.post(f"{http_url}/api/session/start", timeout=30)
            data = r.json()
            if r.status_code == 200 and data.get("status") == "ready":
                print(f"  ready after {time.time() - t0:.1f}s")
                return
            print(f"  not ready yet: {data}")
        except Exception as e:
            print(f"  warmup call failed: {e}")
        time.sleep(3)
    raise SystemExit("Backend/Modal did not warm up in time — aborting.")


async def run_config(
    ws_url: str,
    frames: list[np.ndarray],
    width: int,
    quality: float,
    repeat: int,
    n_frames: int,
    n_warmup: int,
    recv_timeout_s: float,
    writer: "csv.DictWriter",
    csv_file,
) -> list[FrameResult]:
    results: list[FrameResult] = []
    async with websockets.connect(ws_url, max_size=8_000_000) as ws:
        total_to_send = n_warmup + n_frames
        for i in range(total_to_send):
            frame = frames[i % len(frames)]
            jpeg = resize_and_encode(frame, width, quality)

            t_send = time.perf_counter()
            try:
                await ws.send(jpeg)
                raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout_s)
            except Exception as e:
                res = FrameResult(width, quality, repeat, i, len(jpeg), ok=False, error=str(e)[:200])
                if i >= n_warmup:
                    results.append(res)
                    writer.writerow(res.__dict__)
                    csv_file.flush()
                continue
            t_recv = time.perf_counter()

            client_e2e_ms = (t_recv - t_send) * 1000
            try:
                data = json.loads(raw)
            except Exception:
                data = {}

            if "error" in data:
                res = FrameResult(width, quality, repeat, i, len(jpeg), ok=False, error=str(data["error"])[:200])
            else:
                boxes = data.get("boxes", [])
                timing = data.get("timing", {})
                total_ms = timing.get("total_ms", 0)
                res = FrameResult(
                    width=width, quality=quality, repeat=repeat, frame_idx=i,
                    payload_bytes=len(jpeg), ok=True,
                    client_e2e_ms=round(client_e2e_ms, 1),
                    recv_ms=timing.get("recv_ms", 0),
                    modal_ms=timing.get("modal_ms", 0),
                    total_ms=total_ms,
                    network_ms=round(client_e2e_ms - total_ms, 1),
                    num_boxes=len(boxes),
                )

            if i >= n_warmup:  # discard warmup frames from results
                results.append(res)
                writer.writerow(res.__dict__)
                csv_file.flush()

            if (i + 1) % 20 == 0:
                print(f"    [{width}px q{quality} rep{repeat}] {i + 1}/{total_to_send} frames sent")

    return results


def summarize(all_results: list[FrameResult], target_ms: float) -> list[dict]:
    by_config: dict[tuple[int, float], list[FrameResult]] = {}
    for r in all_results:
        by_config.setdefault((r.width, r.quality), []).append(r)

    rows = []
    for (width, quality), rs in sorted(by_config.items()):
        ok = [r for r in rs if r.ok]
        fail = len(rs) - len(ok)
        if not ok:
            rows.append({"width": width, "quality": quality, "frames": len(rs), "failed": fail,
                         "note": "all frames failed"})
            continue
        e2e = [r.client_e2e_ms for r in ok]
        modal = [r.modal_ms for r in ok]
        net = [r.network_ms for r in ok]
        payload = [r.payload_bytes for r in ok]
        mean_e2e = statistics.mean(e2e)
        rows.append({
            "width": width,
            "quality": quality,
            "frames": len(rs),
            "failed": fail,
            "payload_kb_mean": round(statistics.mean(payload) / 1024, 1),
            "e2e_mean_ms": round(mean_e2e, 1),
            "e2e_p50_ms": round(statistics.median(e2e), 1),
            "e2e_p90_ms": round(float(np.percentile(e2e, 90)), 1),
            "e2e_p95_ms": round(float(np.percentile(e2e, 95)), 1),
            "modal_mean_ms": round(statistics.mean(modal), 1),
            "network_mean_ms": round(statistics.mean(net), 1),
            "achievable_fps": round(1000 / mean_e2e, 2),
            "meets_target": mean_e2e <= target_ms,
        })
    return rows


def print_summary(rows: list[dict], target_ms: float) -> None:
    print("\n" + "=" * 100)
    print(f"SUMMARY (target: mean end-to-end latency <= {target_ms} ms)")
    print("=" * 100)
    header = f"{'width':>6} {'qual':>5} {'frames':>7} {'fail':>5} {'kb':>7} {'e2e_mean':>9} {'e2e_p50':>8} {'e2e_p90':>8} {'e2e_p95':>8} {'modal':>7} {'net':>6} {'fps':>6}  pass"
    print(header)
    for r in rows:
        if "note" in r:
            print(f"{r['width']:>6} {r['quality']:>5} {r['frames']:>7} {r['failed']:>5}  {r['note']}")
            continue
        mark = "YES" if r["meets_target"] else "no"
        print(
            f"{r['width']:>6} {r['quality']:>5} {r['frames']:>7} {r['failed']:>5} "
            f"{r['payload_kb_mean']:>7} {r['e2e_mean_ms']:>9} {r['e2e_p50_ms']:>8} "
            f"{r['e2e_p90_ms']:>8} {r['e2e_p95_ms']:>8} {r['modal_mean_ms']:>7} "
            f"{r['network_mean_ms']:>6} {r['achievable_fps']:>6}  {mark}"
        )
    print("=" * 100)


async def main_async(args: argparse.Namespace) -> None:
    widths = [int(w) for w in args.widths.split(",")]
    qualities = [float(q) for q in args.qualities.split(",")]

    frames = load_frames(Path(args.video))
    print(f"Loaded {len(frames)} frames from {args.video}")

    parsed = urlparse(args.http_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/api/ws/infer"

    if not args.skip_warmup:
        warmup_backend(args.http_url)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"latency_{ts}.csv"
    summary_path = RESULTS_DIR / f"latency_{ts}_summary.csv"

    fieldnames = list(FrameResult.__dataclass_fields__.keys())
    all_results: list[FrameResult] = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_configs = len(widths) * len(qualities) * args.repeats
        done = 0
        try:
            for rep in range(args.repeats):
                for width in widths:
                    for quality in qualities:
                        done += 1
                        print(f"\n[{done}/{total_configs}] width={width} quality={quality} repeat={rep}")
                        res = await run_config(
                            ws_url, frames, width, quality, rep,
                            n_frames=args.frames, n_warmup=args.warmup_frames,
                            recv_timeout_s=args.recv_timeout, writer=writer, csv_file=f,
                        )
                        all_results.extend(res)
        except KeyboardInterrupt:
            print("\nInterrupted — writing partial results...")

    print(f"\nPer-frame results written to {csv_path}")

    rows = summarize(all_results, args.target_ms)
    with open(summary_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Summary written to {summary_path}")

    print_summary(rows, args.target_ms)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--http-url", default="http://localhost:8000", help="Backend base URL (http/https)")
    p.add_argument("--video", default="data/samples/videos/test_polyp_seq2.mp4", help="Source video to replay frames from")
    p.add_argument("--widths", default="160,224,320,480,640", help="Comma-separated resize widths (px)")
    p.add_argument("--qualities", default="0.6,0.85", help="Comma-separated JPEG qualities (0-1)")
    p.add_argument("--frames", type=int, default=60, help="Timed frames per config")
    p.add_argument("--warmup-frames", type=int, default=5, help="Untimed frames sent before recording, per config")
    p.add_argument("--repeats", type=int, default=1, help="Repeat the whole matrix N times")
    p.add_argument("--recv-timeout", type=float, default=10.0, help="Per-frame response timeout (s)")
    p.add_argument("--target-ms", type=float, default=200.0, help="Target mean end-to-end latency for pass/fail marking")
    p.add_argument("--skip-warmup", action="store_true", help="Skip the /api/session/start warmup call")
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
