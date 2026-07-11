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

## Part 2: is it GPU compute or invocation overhead?

Added `PolypDetector.infer_frame_bench` (`inference/app.py`, benchmark-only, doesn't touch the
production `infer_frame`/`infer_video` paths) — same work as `infer_frame` but times JPEG decode
and the GPU forward pass *inside the container* and returns those alongside the boxes. Then
`experiments/modal_rpc_bench.py` calls it directly through the Modal SDK (`.remote.aio()`, same
pattern as `backend/services/modal_client.py`), bypassing FastAPI/WebSocket entirely, so it
isolates the backend-server↔Modal leg. 30 timed calls per width, run from the server.

| width | rpc mean (ms) | gpu mean (ms) | decode mean (ms) | overhead mean (ms) | overhead % |
|---|---|---|---|---|---|
| 160 | 255.8 | 11.7 | 0.2 | 243.9 | 95.3% |
| 320 | 272.1 | 11.7 | 0.4 | 260.0 | 95.6% |
| 640 | 255.7 | 10.7 | 1.1 | 243.9 | 95.4% |

Raw data: `results/modal_rpc_bench_20260711_095610.csv` / `..._summary.csv`.

**GPU inference is ~11ms. Decode is ~1ms. 95%+ of the latency (~245-260ms) is invocation
overhead** — confirmed not payload size (flat across 160→640px, matching part 1).

Checked whether that overhead is physical network distance: the backend server is on Hetzner in
Helsinki, Finland; `api.modal.com` resolves to an AWS `us-east`-range IP (`54.163.156.253`).
Raw TCP connect time from the server to `api.modal.com:443` is **~107ms** — a transatlantic hop.
ICMP is blocked so no clean ping RTT, but the TCP-connect time alone accounts for roughly 40% of
the total round trip; the rest is TLS + however many round trips Modal's RPC protocol needs per
invocation (control-plane dispatch to the container, then the result).

**Conclusion: the ~250ms floor is mostly the Helsinki↔US network distance, not GPU work or
payload size.** Resizing frames (part 1) was never going to fix this — there was nothing to
trim; the frame data isn't the bottleneck.

### Options, not yet tried

- **Move the backend closer to Modal's region** (or find out if Modal can run this GPU function
  in an EU region) — this is the lever with the biggest expected payoff, since it directly
  attacks the transatlantic RTT. Needs checking Modal's docs/support for region selection, and
  is an infra/hosting decision, not a code change.
- **Reduce round trips per call** if Modal's protocol allows it (e.g. a persistent
  stream/connection instead of one invocation per frame) — would cut the multiplier on the
  network RTT even without moving anything.
- **Accept ~250ms as a floor** and design the UI around it (e.g. don't chase sub-200ms
  motion-to-photon; smooth/interpolate box positions between updates instead) if moving compute
  isn't practical.
- Pipelining (frame N+1 sent before frame N's result returns) raises achievable fps but does
  **not** reduce the lag of any single frame — doesn't address the "boxes lag when the camera
  moves" complaint by itself, only throughput.
