"use client";

import { useEffect, useState } from "react";

const STEPS = [
  { at: 0,     text: "Requesting A100 GPU on Modal..." },
  { at: 5000,  text: "Container provisioning..." },
  { at: 12000, text: "Loading runtime environment..." },
  { at: 22000, text: "Downloading model weights (50 MB)..." },
  { at: 38000, text: "Loading YOLOv5 into GPU memory..." },
  { at: 52000, text: "Warming up inference pipeline..." },
];

export default function WarmupPanel() {
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    const timers = STEPS.map(({ at, text }) =>
      setTimeout(() => setLogs((prev) => [...prev, text]), at)
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  const progress = Math.min(90, Math.round((logs.length / STEPS.length) * 100));

  return (
    <div className="space-y-5 py-4">
      <div>
        <div className="flex justify-between text-sm text-gray-400 mb-2">
          <span>Starting GPU session</span>
          <span>{progress}%</span>
        </div>
        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-1000 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 font-mono text-xs text-green-400 h-44 overflow-y-auto">
        {logs.map((line, i) => (
          <div key={i} className="leading-6">
            <span className="text-gray-600 select-none mr-2">
              {String(i + 1).padStart(2, "0")}
            </span>
            {line}
          </div>
        ))}
        {logs.length > 0 && (
          <span className="animate-pulse text-blue-400">▋</span>
        )}
      </div>

      <p className="text-xs text-gray-600 text-center">
        Cold start ~60s · Warm starts are instant · Stop with{" "}
        <code className="text-gray-500">modal app stop polyp-detection</code>
      </p>
    </div>
  );
}
