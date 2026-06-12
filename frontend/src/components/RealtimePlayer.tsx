"use client";

import { useEffect, useRef, useState } from "react";
import DemoVideoPicker from "./DemoVideoPicker";

const API_WS = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001").replace(/^http/, "ws");
const INFER_TIMEOUT_MS = 6000;
// Resize frames to this width before sending — faster inference, smaller payload
const INFER_WIDTH = 320;

interface Box { bbox: [number, number, number, number]; conf: number; }
interface Timing { recv_ms: number; modal_ms: number; total_ms: number; }

export default function RealtimePlayer({ onStop, wsPath = "/api/ws/infer" }: { onStop: () => void; wsPath?: string }) {
  const WS_URL = `${API_WS}${wsPath}`;
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef     = useRef<WebSocket | null>(null);
  const rafRef    = useRef<number>(0);
  const scanRef   = useRef(false); // scan loop running?

  // Pending response promise resolver — one in-flight request at a time
  const pendingRef = useRef<((v: { boxes: Box[]; timing: Timing } | null) => void) | null>(null);

  const [videoUrl, setVideoUrl]   = useState<string | null>(null);
  const [tab, setTab]             = useState<"upload" | "demo">("upload");
  const [dragging, setDragging]   = useState(false);
  const [wsStatus, setWsStatus]   = useState<string>("connecting");
  const [polyp, setPolyp]         = useState(false);
  const [boxesState, setBoxes]    = useState<Box[]>([]);
  const [stats, setStats]         = useState({ sent: 0, received: 0, avgMs: 0 });
  const [lastError, setLastError] = useState("");
  const msHistory = useRef<number[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  // Canvas draw loop — full framerate
  useEffect(() => {
    function draw() {
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) { rafRef.current = requestAnimationFrame(draw); return; }

      const w = video.videoWidth  || 560;
      const h = video.videoHeight || 480;
      if (canvas.width !== w)  canvas.width  = w;
      if (canvas.height !== h) canvas.height = h;

      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, w, h);

      const scale = w / INFER_WIDTH;
      for (const det of boxesRef.current) {
        let [x1, y1, x2, y2] = det.bbox;
        // Scale boxes back to display resolution
        x1 *= scale; y1 *= scale; x2 *= scale; y2 *= scale;
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
      rafRef.current = requestAnimationFrame(draw);
    }
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  // Shared boxes ref for canvas — synced via setBoxes + ref
  const boxesRef = useRef<Box[]>([]);
  function updateBoxes(b: Box[]) { boxesRef.current = b; setBoxes(b); setPolyp(b.length > 0); }

  // Frame-step scan loop — send frame → wait for result → advance 1 frame → repeat
  async function startScan(video: HTMLVideoElement) {
    if (scanRef.current) return;
    scanRef.current = true;
    const fps = 25; // video was made at 25fps
    const frameDuration = 1 / fps;

    video.pause();
    video.currentTime = 0;

    while (scanRef.current) {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) break;

      // Capture current frame at reduced size
      const scale = INFER_WIDTH / video.videoWidth;
      const capW  = INFER_WIDTH;
      const capH  = Math.round(video.videoHeight * scale);
      const cap   = document.createElement("canvas");
      cap.width   = capW;
      cap.height  = capH;
      cap.getContext("2d")!.drawImage(video, 0, 0, capW, capH);

      const blob: Blob = await new Promise((res) => cap.toBlob((b) => res(b!), "image/jpeg", 0.85));
      const buf = await blob.arrayBuffer();

      // Send and wait for response (with timeout)
      const result = await new Promise<{ boxes: Box[]; timing: Timing } | null>((resolve) => {
        pendingRef.current = resolve;
        ws.send(buf);
        setStats((s) => ({ ...s, sent: s.sent + 1 }));
        setTimeout(() => {
          if (pendingRef.current === resolve) { pendingRef.current = null; resolve(null); }
        }, INFER_TIMEOUT_MS);
      });

      if (result) updateBoxes(result.boxes);

      // Advance video by 1 frame
      const next = video.currentTime + frameDuration;
      if (next >= video.duration) {
        video.currentTime = 0; // loop back
      } else {
        video.currentTime = next;
      }

      // Let the browser render the new frame before capturing again
      await new Promise<void>((res) => requestAnimationFrame(() => res()));
    }

    scanRef.current = false;
  }

  function handleVideoLoad() {
    const video = videoRef.current;
    if (!video) return;
    startScan(video);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) { scanRef.current = false; setVideoUrl(URL.createObjectURL(file)); }
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) { scanRef.current = false; setVideoUrl(URL.createObjectURL(file)); }
  }

  function handleDemoSelect(filename: string) {
    scanRef.current = false;
    setVideoUrl(`/demos/${filename}`);
  }

  const wsOk = wsStatus === "open";

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
        <button onClick={onStop} className="text-sm text-red-400 hover:text-red-300 transition-colors">Stop</button>
      </div>

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

      {!videoUrl && (
        <div className="space-y-4">
          {/* Tab switcher */}
          <div className="flex gap-1 border-b border-gray-800">
            {(["upload", "demo"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                  tab === t
                    ? "border-green-500 text-white"
                    : "border-transparent text-gray-500 hover:text-gray-300"
                }`}
              >
                {t === "upload" ? "Upload video" : "Try a demo"}
              </button>
            ))}
          </div>

          {tab === "upload" && (
            <div
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-20 text-center cursor-pointer transition-colors ${
                dragging ? "border-green-400 bg-green-950/20" : "border-gray-700 hover:border-gray-500"
              }`}
            >
              <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={handleFile} />
              <p className="text-gray-200 text-lg">Drop a colonoscopy video here</p>
              <p className="text-gray-500 text-sm mt-2">Steps frame-by-frame — boxes always sync to current frame</p>
            </div>
          )}

          {tab === "demo" && (
            <DemoVideoPicker onSelect={handleDemoSelect} />
          )}
        </div>
      )}

      {videoUrl && (
        <>
          <div className="relative w-full rounded-xl overflow-hidden border border-gray-800 bg-black"
            style={{ aspectRatio: "560/480" }}>
            <video
              ref={videoRef}
              src={videoUrl}
              muted
              onCanPlay={handleVideoLoad}
              className="absolute inset-0 w-full h-full"
            />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
          </div>
          <button
            onClick={() => { scanRef.current = false; setVideoUrl(null); updateBoxes([]); }}
            className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
          >
            ← Load different video
          </button>
        </>
      )}

      <p className="text-xs text-gray-600">
        Frames scaled to {INFER_WIDTH}px before sending · step-through synced to inference speed
      </p>
    </div>
  );
}
