# Live-inference latency benchmark

Run: `experiments/latency_bench.py` — see that file's docstring for the full protocol.

**Setup:** ran from the server itself (`arman@call2fly.ai`) against the live backend at
`127.0.0.1:19000`, replaying `frontend/public/demos/test_polyp_seq2.mp4` (206 frames) frame by
frame — one frame in flight at a time, exactly like `RealtimePlayer.tsx` does today (send →
wait for boxes → send next). Swept resize width × JPEG quality, 60 timed frames per config
(+5 discarded warm-up frames), Modal container pre-warmed via `/api/session/start` first.

Raw data: `results/latency_20260711_092306.csv` (per-frame) and
`results/latency_20260711_092306_summary.csv` (per-config).

## Results

| width | quality | payload (KB) | e2e mean (ms) | e2e p50 | e2e p90 | e2e p95 | modal (ms) | network (ms) | fps @ 1-in-flight |
|---|---|---|---|---|---|---|---|---|---|
| 160 | 0.60 | 3.1  | 255.3 | 249.9 | 279.4 | 285.8 | 253.2 | 2.0 | 3.92 |
| 160 | 0.85 | 5.4  | 250.4 | 246.8 | 260.6 | 276.1 | 248.3 | 2.1 | 3.99 |
| 224 | 0.60 | 4.8  | 254.0 | 248.9 | 264.8 | 277.1 | 252.1 | 1.9 | 3.94 |
| 224 | 0.85 | 8.7  | 260.0 | 251.4 | 280.6 | 330.3 | 257.5 | 2.5 | 3.85 |
| 320 | 0.60 | 8.2  | 256.9 | 250.6 | 274.1 | 293.2 | 254.2 | 2.7 | 3.89 |
| 320 | 0.85 | 15.1 | 259.6 | 253.3 | 267.6 | 282.3 | 255.9 | 3.7 | 3.85 |
| 480 | 0.60 | 15.4 | 270.9 | 254.8 | 290.4 | 363.6 | 267.8 | 3.1 | 3.69 |
| 480 | 0.85 | 28.3 | 260.3 | 255.2 | 278.4 | 290.4 | 255.7 | 4.5 | 3.84 |
| 640 | 0.60 | 24.0 | 261.1 | 252.6 | 276.1 | 300.1 | 257.2 | 3.9 | 3.83 |
| 640 | 0.85 | 43.2 | 264.6 | 258.6 | 293.9 | 319.4 | 259.0 | 5.6 | 3.78 |

*(0 failed frames in every config.)*

## Key finding

**Resizing barely moves end-to-end latency.** Going from 160px/q0.6 (3 KB payload) all the way
up to 640px/q0.85 (43 KB — a 14x larger payload) only adds ~10ms to the mean round trip
(250ms → 265ms). `network_ms` (client↔backend transport) stays under 6ms at every size, and
this run didn't even cross a real network — it hit `127.0.0.1` on the same box.

The ~250-260ms is almost entirely `modal_ms`: the round trip to the Modal GPU function itself
(RPC/invocation overhead + the actual forward pass), which is roughly **fixed regardless of
image size**. So the lag isn't a bandwidth problem — it's the per-call cost of hitting Modal.
Shrinking the frame won't fix it.

## What this means for a target latency

None of the tested configs get anywhere close to a "feels responsive while panning" range
(commonly cited as ~100-150ms motion-to-photon, ~200ms as the outer edge of tolerable). Every
config here lands at ~250-270ms mean, ~270-360ms at p95.

Since resizing isn't the lever, options worth benchmarking next (not yet built/tested):
- Measure how much of `modal_ms` is fixed RPC overhead vs. actual GPU inference (e.g. call the
  Modal function directly and time just `model()` inside `infer_frame`, without changing the
  request pattern), to see if there's a fixed floor we can't resize our way under.
- Decouple capture from the wait: instead of strict request→wait→next-frame, allow 2 frames
  in flight (send frame N+1 before frame N's result comes back) so the *displayed* frame rate
  isn't gated by round-trip latency, even though any single frame's latency stays ~250ms.
- Check whether the Modal container is scaling to zero between calls despite the warm-up
  (`scaledown_window=60`) — a cold GPU container would show up as occasional multi-second
  spikes in the p95/p99, which we didn't see here since frames were sent continuously.

## Real-world caveat

This run measured backend↔Modal latency in isolation (script ran on the same server as the
backend). The actual browser↔backend leg (clinic device → internet → server) is a separate,
additive cost not captured here — worth a second pass once the live-camera UI exists, run from
a machine on the clinic's actual network path.
