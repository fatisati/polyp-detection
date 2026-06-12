"""
Local inference wrapper — use for quick tests without deploying to Modal.
Run:  python -m inference.model path/to/clip.mp4 out.mp4
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

YOLO_WEIGHTS = "yolov8s.pt"  # swap for fine-tuned weights when available


class PolypDetectorLocal:
    def __init__(self, weights: str = YOLO_WEIGHTS):
        self.model = YOLO(weights)

    def infer_frame(self, frame: np.ndarray, conf: float = 0.3) -> dict:
        results = self.model(frame, conf=conf, verbose=False)[0]
        detections = [
            {"bbox": box.xyxy[0].tolist(), "confidence": float(box.conf[0])}
            for box in results.boxes
        ]
        return {"detections": detections, "annotated": results.plot()}

    def infer_video(self, in_path: str, out_path: str, conf: float = 0.3) -> dict:
        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        frames_with_polyp = 0
        for _ in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=conf, verbose=False)[0]
            if len(results.boxes):
                frames_with_polyp += 1
            writer.write(results.plot())

        cap.release()
        writer.release()

        return {
            "total_frames": total,
            "frames_with_detection": frames_with_polyp,
            "detection_rate": frames_with_polyp / total if total else 0,
        }


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
    detector = PolypDetectorLocal()
    stats = detector.infer_video(in_path, out_path)
    print(stats)
