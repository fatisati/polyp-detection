"use client";

type Stage = "uploading" | "gpu-warmup" | "inferring" | string;

interface Props {
  stage: Stage;
  logs: string[];
}

const LABEL: Record<string, string> = {
  uploading: "Uploading",
  "gpu-warmup": "Starting GPU",
  inferring: "Running inference",
};

const PROGRESS: Record<string, number> = {
  uploading: 10,
  "gpu-warmup": 40,
  inferring: 85,
};

export default function GpuStatusBar({ stage, logs }: Props) {
  const progress = PROGRESS[stage] ?? 0;
  const label = LABEL[stage] ?? stage;

  return (
    <div className="space-y-4">
      <div>
        <div className="flex justify-between text-sm text-gray-400 mb-1.5">
          <span>{label}</span>
          <span>{progress}%</span>
        </div>
        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-700 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 font-mono text-xs text-green-400 h-44 overflow-y-auto">
        {logs.map((line, i) => (
          <div key={i} className="leading-5">
            <span className="text-gray-600 select-none mr-2">
              {String(i + 1).padStart(2, "0")}
            </span>
            {line}
          </div>
        ))}
        <span className="animate-pulse">▋</span>
      </div>
    </div>
  );
}
