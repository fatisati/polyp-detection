"use client";

import { useEffect, useRef, useState } from "react";

const API_WS = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001").replace(/^http/, "ws");
const INFER_TIMEOUT_MS = 6000;
// Resize frames to this width before sending — faster inference, smaller payload
const INFER_WIDTH = 320;

interface Box { bbox: [number, number, number, number]; conf: number; }
interface Timing { recv_ms: number; modal_ms: number; total_ms: number; }

export default function LiveCameraPlayer({ onStop, wsPath = "/api/ws/infer" }: { onStop: () => void; wsPath?: string }) {
  const WS_URL = `${API_WS}${wsPath}`;
  const videoRef    = useRef<HTMLVideoElement>(null);
  const analyzedRef = useRef<HTMLCanvasElement>(null); // last frame actually sent to the model, with boxes burned on
  const wsRef       = useRef<WebSocket | null>(null);
  const scanRef     = useRef(false); // capture loop running?
  const streamRef   = useRef<MediaStream | null>(null);

  // Pending response promise resolver — one in-flight request at a time
  const pendingRef = useRef<((v: { boxes: Box[]; timing: Timing } | null) => void) | null>(null);

  const [devices, setDevices]             = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedId] = useState("");
  const [permission, setPermission]       = useState<"idle" | "requesting" | "granted" | "denied">("idle");
  const [streaming, setStreaming]         = useState(false);
  const [wsStatus, setWsStatus]           = useState<string>("connecting");
  const [polyp, setPolyp]                 = useState(false);
  const [stats, setStats]                 = useState({ sent: 0, received: 0, avgMs: 0 });
  const [lastError, setLastError]         = useState("");
  const msHistory = useRef<number[]>([]);

  const insecure = typeof window !== "undefined" && !window.isSecureContext;

  // WebSocket — connect once on mount
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen  = () => setWsStatus("open");
    ws.onerror = () => setWsStatus("error");
    ws.onclose = (e) => { setWsStatus(`closed (${e.code})`); pendingRef.current?.(null); };

    ws.onmessage = (e) => {
      let data: unknown;
      try { data = JSON.parse(e.data); } catch { return; }

      if (data && typeof data === "object" && "error" in data) {
        const err = (data as { error: string }).error;
        setLastError(err);
        pendingRef.current?.(null);
        pendingRef.current = null;
        return;
      }

      const { boxes, timing } = data as { boxes: Box[]; timing: Timing };
      msHistory.current.push(timing.modal_ms);
      if (msHistory.current.length > 10) msHistory.current.shift();
      const avg = Math.round(msHistory.current.reduce((a, b) => a + b, 0) / msHistory.current.length);
      setStats((s) => ({ sent: s.sent, received: s.received + 1, avgMs: avg }));

      pendingRef.current?.({ boxes, timing });
      pendingRef.current = null;
    };

    return () => { ws.close(); scanRef.current = false; };
  }, []);

  // Stop camera tracks on unmount
  useEffect(() => {
    return () => {
      scanRef.current = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Draws the exact frame that was sent to the model, with boxes at native (capture)
  // resolution — pixel-accurate for that frame, unlike overlaying on the live video
  // (which has moved on by the time the result comes back).
  function drawAnalyzedFrame(source: HTMLCanvasElement, boxes: Box[]) {
    const canvas = analyzedRef.current;
    if (!canvas) return;
    if (canvas.width !== source.width)   canvas.width  = source.width;
    if (canvas.height !== source.height) canvas.height = source.height;

    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(source, 0, 0);

    for (const det of boxes) {
      const [x1, y1, x2, y2] = det.bbox;
      ctx.shadowColor = "#39ff14";
      ctx.shadowBlur  = 10;
      ctx.strokeStyle = "#39ff14";
      ctx.lineWidth   = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.shadowBlur  = 0;
      const label = `polyp  ${Math.round(det.conf * 100)}%`;
      ctx.font = "bold 13px monospace";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "#39ff14";
      ctx.fillRect(x1, y1 - 20, tw + 8, 20);
      ctx.fillStyle = "#000";
      ctx.fillText(label, x1 + 4, y1 - 5);
    }
  }

  function updateBoxes(b: Box[]) { setPolyp(b.length > 0); }

  // Live loop — send current frame → wait for result → send next. No seeking:
  // the video plays continuously, we just grab whatever frame is current each time.
  async function startLoop() {
    if (scanRef.current) return;
    scanRef.current = true;

    while (scanRef.current) {
      const ws = wsRef.current;
      const video = videoRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !video || !video.videoWidth) {
        await new Promise<void>((res) => requestAnimationFrame(() => res()));
        continue;
      }

      const scale = INFER_WIDTH / video.videoWidth;
      const capW  = INFER_WIDTH;
      const capH  = Math.round(video.videoHeight * scale);
      const cap   = document.createElement("canvas");
      cap.width   = capW;
      cap.height  = capH;
      cap.getContext("2d")!.drawImage(video, 0, 0, capW, capH);

      const blob: Blob = await new Promise((res) => cap.toBlob((b) => res(b!), "image/jpeg", 0.85));
      const buf = await blob.arrayBuffer();

      const result = await new Promise<{ boxes: Box[]; timing: Timing } | null>((resolve) => {
        pendingRef.current = resolve;
        ws.send(buf);
        setStats((s) => ({ ...s, sent: s.sent + 1 }));
        setTimeout(() => {
          if (pendingRef.current === resolve) { pendingRef.current = null; resolve(null); }
        }, INFER_TIMEOUT_MS);
      });

      // Draw the frame + its boxes together, win or lose (a timeout leaves the last good frame up)
      if (result) {
        updateBoxes(result.boxes);
        drawAnalyzedFrame(cap, result.boxes);
      }
    }
  }

  // First call: unlocks device labels via a throwaway permission prompt, then lists devices.
  async function requestDevices() {
    setPermission("requesting");
    setLastError("");
    try {
      const probe = await navigator.mediaDevices.getUserMedia({ video: true });
      probe.getTracks().forEach((t) => t.stop());

      const all = await navigator.mediaDevices.enumerateDevices();
      const cams = all.filter((d) => d.kind === "videoinput");
      setDevices(cams);
      setPermission("granted");
      if (cams.length > 0) {
        setSelectedId(cams[0].deviceId);
        await startStream(cams[0].deviceId);
      }
    } catch (err: unknown) {
      setPermission("denied");
      setLastError(err instanceof Error ? err.message : String(err));
    }
  }

  async function startStream(deviceId: string) {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    scanRef.current = false;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }
      setStreaming(true);
      startLoop();
    } catch (err: unknown) {
      setLastError(err instanceof Error ? err.message : String(err));
    }
  }

  function handleDeviceChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const id = e.target.value;
    setSelectedId(id);
    startStream(id);
  }

  function stopCamera() {
    scanRef.current = false;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStreaming(false);
    updateBoxes([]);
  }

  const wsOk = wsStatus === "open";
  const deviceSelect = (
    <select
      value={selectedDeviceId}
      onChange={handleDeviceChange}
      className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
    >
      {devices.map((d) => (
        <option key={d.deviceId} value={d.deviceId}>
          {d.label || `Camera ${d.deviceId.slice(0, 6)}`}
        </option>
      ))}
    </select>
  );

  return (
    <div className="space-y-3">
      {/* Status bar */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-3">
          <span className={`flex items-center gap-1.5 ${wsOk ? "text-green-400" : "text-yellow-400"}`}>
            <span className={`w-2 h-2 rounded-full inline-block ${wsOk ? "bg-green-400 animate-pulse" : "bg-yellow-400"}`} />
            {wsOk ? "connected" : wsStatus}
          </span>
          {polyp && <span className="text-[#39ff14] font-medium animate-pulse">Polyp detected</span>}
        </div>
        <button onClick={() => { stopCamera(); onStop(); }} className="text-sm text-red-400 hover:text-red-300 transition-colors">Stop</button>
      </div>

      {insecure && (
        <div className="bg-yellow-950 border border-yellow-800 rounded-lg px-3 py-2 text-yellow-300 text-xs">
          This page isn&apos;t served over HTTPS (or localhost) — browsers block camera access on insecure origins.
          Open it via <code>https://</code> or <code>localhost</code> for the camera to work.
        </div>
      )}

      {/* Debug panel */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 font-mono text-xs grid grid-cols-2 gap-x-6 gap-y-0.5">
        <span className="text-gray-500">Frames sent</span>
        <span className="text-white">{stats.sent}</span>
        <span className="text-gray-500">Responses back</span>
        <span className="text-white">{stats.received}</span>
        <span className="text-gray-500">Modal latency (avg)</span>
        <span className={stats.avgMs > 800 ? "text-red-400" : "text-green-400"}>
          {stats.avgMs > 0 ? `${stats.avgMs} ms` : "—"}
        </span>
        {lastError && <>
          <span className="text-gray-500">Error</span>
          <span className="text-red-400 truncate">{lastError}</span>
        </>}
      </div>

      {!streaming && (
        <div className="space-y-4 text-center py-16 border-2 border-dashed border-gray-700 rounded-xl">
          {permission !== "granted" ? (
            <>
              <p className="text-gray-200 text-lg">Connect a camera</p>
              <p className="text-gray-500 text-sm max-w-sm mx-auto px-4">
                Laptop webcam, phone camera (if this page is opened on the phone itself), or a
                USB/HDMI capture card — any of them show up below once you grant camera access.
              </p>
              <button
                onClick={requestDevices}
                className="px-6 py-2.5 bg-purple-600 hover:bg-purple-500 rounded-lg text-white font-medium transition-colors"
              >
                {permission === "requesting" ? "Requesting access..." : "Choose a camera"}
              </button>
            </>
          ) : (
            <p className="text-gray-500 text-sm">No active stream — pick a device below.</p>
          )}
          {devices.length > 0 && <div>{deviceSelect}</div>}
        </div>
      )}

      {streaming && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <p className="text-xs text-gray-500 uppercase tracking-wide">Live · no lag</p>
              <div className="relative w-full rounded-xl overflow-hidden border border-gray-800 bg-black"
                style={{ aspectRatio: "560/480" }}>
                <video ref={videoRef} muted playsInline className="absolute inset-0 w-full h-full object-cover" />
              </div>
            </div>
            <div className="space-y-1.5">
              <p className="text-xs text-gray-500 uppercase tracking-wide">
                Detected · ~{stats.avgMs || 250}ms behind live
              </p>
              <div className="relative w-full rounded-xl overflow-hidden border border-gray-800 bg-black"
                style={{ aspectRatio: "560/480" }}>
                <canvas ref={analyzedRef} className="absolute inset-0 w-full h-full object-contain" />
              </div>
            </div>
          </div>

          {devices.length > 1 && deviceSelect}

          <button onClick={stopCamera} className="text-sm text-gray-500 hover:text-gray-300 transition-colors">
            ← Disconnect camera
          </button>
        </>
      )}

      <p className="text-xs text-gray-600">
        Frames scaled to {INFER_WIDTH}px before sending · one frame in flight at a time · ~{stats.avgMs || 250}ms round trip per frame
      </p>
    </div>
  );
}
