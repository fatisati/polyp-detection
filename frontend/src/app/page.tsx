"use client";

import { useRef, useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import WarmupPanel from "@/components/WarmupPanel";
import VideoPlayer from "@/components/VideoPlayer";
import RealtimePlayer from "@/components/RealtimePlayer";

type Stage = "idle" | "warming" | "ready" | "processing" | "done";

interface Detection {
  bbox: [number, number, number, number];
  conf: number;
}

interface InferResult {
  fps: number;
  width: number;
  height: number;
  frames: Detection[][];
}

interface GtData {
  width: number;
  height: number;
  frames: { bbox: [number, number, number, number] }[][];
}

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

export default function Home() {
  const [stage, setStage] = useState<Stage>("idle");
  const [mode, setMode] = useState<"upload" | "realtime" | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [result, setResult] = useState<InferResult | null>(null);
  const [groundTruth, setGroundTruth] = useState<GtData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const gtInputRef = useRef<HTMLInputElement>(null);

  async function handleStop() {
    setStopping(true);
    try {
      await fetch(`${API}/api/session/stop`, { method: "POST" });
    } finally {
      setStopping(false);
      setStage("idle");
      setResult(null);
      setVideoUrl(null);
      setError(null);
      setGroundTruth(null);
    }
  }

  async function handleStart() {
    setStage("warming");
    setError(null);
    try {
      const res = await fetch(`${API}/api/session/start`, { method: "POST" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Backend error ${res.status}: ${text}`);
      }
      setStage("ready");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("fetch") || msg.includes("Failed to fetch")) {
        setError(`Cannot reach backend at ${API}. Is uvicorn running?`);
      } else {
        setError(msg);
      }
      setStage("idle");
    }
  }

  async function handleUpload(file: File) {
    setError(null);
    setGroundTruth(null);
    const localUrl = URL.createObjectURL(file);
    setVideoUrl(localUrl);
    setStage("processing");

    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${API}/api/infer-video`, {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      setError("Inference failed. Check backend logs.");
      setStage("ready");
      return;
    }

    const data: InferResult = await res.json();
    setResult(data);
    setStage("done");
  }

  async function handleGtFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setGroundTruth(JSON.parse(text) as GtData);
    // reset input so same file can be re-selected
    e.target.value = "";
  }

  function resetToUpload() {
    setStage("ready");
    setMode(null);
    setResult(null);
    setVideoUrl(null);
    setGroundTruth(null);
  }

  return (
    <main className="min-h-screen bg-gray-950 text-white p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-1">Polyp Detection AI</h1>
      <p className="text-gray-500 text-sm mb-10">
        Real-time colonoscopy polyp detection · YOLOv5 · Kvasir-SEG · mAP50 0.93
      </p>

      {stage === "idle" && (
        <div className="flex flex-col items-center justify-center py-24 gap-6">
          <p className="text-gray-400 text-center max-w-sm">
            Starts a GPU session on Modal. First start takes ~60 seconds to load the model.
          </p>
          <button
            onClick={handleStart}
            className="px-8 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl text-white font-medium text-lg transition-colors"
          >
            Start Session
          </button>
        </div>
      )}

      {stage === "warming" && <WarmupPanel />}

      {stage === "ready" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-green-400 text-sm">
              <span className="w-2 h-2 rounded-full bg-green-400 inline-block animate-pulse" />
              GPU ready · Model loaded · A100 active
            </div>
            <button
              onClick={handleStop}
              disabled={stopping}
              className="text-sm text-red-400 hover:text-red-300 transition-colors disabled:opacity-50"
            >
              {stopping ? "Ending..." : "End Session"}
            </button>
          </div>

          {mode === null && (
            <div className="grid grid-cols-2 gap-4 pt-4">
              <button
                onClick={() => setMode("upload")}
                className="flex flex-col items-center gap-3 p-8 rounded-xl border-2 border-gray-700 hover:border-blue-500 hover:bg-blue-950/10 transition-colors text-center"
              >
                <span className="text-3xl">🎬</span>
                <span className="text-white font-medium">Upload Video</span>
                <span className="text-gray-500 text-sm">Analyse a recorded colonoscopy clip</span>
              </button>
              <button
                onClick={() => setMode("realtime")}
                className="flex flex-col items-center gap-3 p-8 rounded-xl border-2 border-gray-700 hover:border-green-500 hover:bg-green-950/10 transition-colors text-center"
              >
                <span className="text-3xl">📷</span>
                <span className="text-white font-medium">Real-time</span>
                <span className="text-gray-500 text-sm">Frame-by-frame · ~600ms/frame</span>
              </button>
            </div>
          )}

          {mode === "upload" && (
            <div className="space-y-3">
              <button onClick={() => setMode(null)} className="text-sm text-gray-500 hover:text-gray-300 transition-colors">
                ← Back
              </button>
              <UploadPanel onUpload={handleUpload} />
            </div>
          )}

          {mode === "realtime" && (
            <div className="space-y-3">
              <button onClick={() => setMode(null)} className="text-sm text-gray-500 hover:text-gray-300 transition-colors">
                ← Back
              </button>
              <RealtimePlayer onStop={() => setMode(null)} />
            </div>
          )}
        </div>
      )}

      {stage === "processing" && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-yellow-400 text-sm animate-pulse">
            <span className="w-2 h-2 rounded-full bg-yellow-400 inline-block" />
            Running inference on Modal...
          </div>
          {videoUrl && (
            <video src={videoUrl} autoPlay loop muted
              className="w-full rounded-xl border border-gray-800 bg-black opacity-40" />
          )}
        </div>
      )}

      {stage === "done" && videoUrl && result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-green-400 text-sm">
              <span className="w-2 h-2 rounded-full bg-green-400 inline-block" />
              Inference complete · {result.frames.length} frames · {result.fps.toFixed(0)} fps
            </div>
            <div className="flex gap-4">
              <button
                onClick={resetToUpload}
                className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
              >
                Try another video
              </button>
              <button
                onClick={handleStop}
                disabled={stopping}
                className="text-sm text-red-400 hover:text-red-300 transition-colors disabled:opacity-50"
              >
                {stopping ? "Ending..." : "End Session"}
              </button>
            </div>
          </div>

          <VideoPlayer
            src={videoUrl}
            detections={result.frames}
            fps={result.fps}
            width={result.width}
            height={result.height}
            groundTruth={groundTruth?.frames}
          />

          {/* Ground truth loader */}
          <div className={`rounded-xl border px-4 py-3 flex items-center justify-between transition-colors ${
            groundTruth ? "border-cyan-800 bg-cyan-950/20" : "border-gray-700 bg-gray-900/50"
          }`}>
            {groundTruth ? (
              <>
                <span className="text-sm text-cyan-400">Ground truth loaded · cyan boxes active</span>
                <button
                  onClick={() => setGroundTruth(null)}
                  className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
                >
                  Remove
                </button>
              </>
            ) : (
              <>
                <span className="text-sm text-gray-400">
                  Want to compare with ground truth?
                </span>
                <button
                  onClick={() => gtInputRef.current?.click()}
                  className="text-sm px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-white transition-colors"
                >
                  Load GT JSON
                </button>
              </>
            )}
          </div>
          <input ref={gtInputRef} type="file" accept=".json" className="hidden" onChange={handleGtFile} />

          <button
            onClick={resetToUpload}
            className="w-full py-3 bg-gray-800 hover:bg-gray-700 rounded-xl text-white font-medium transition-colors"
          >
            Upload another video
          </button>
        </div>
      )}

      {error && (
        <div className="mt-4 bg-red-950 border border-red-800 rounded-lg px-4 py-3 text-red-300 text-sm font-mono">
          {error}
        </div>
      )}
    </main>
  );
}
