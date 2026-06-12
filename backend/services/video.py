"""OpenCV helpers for frame extraction and video assembly."""

from pathlib import Path
import cv2
import numpy as np


def get_video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def extract_frames(path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and len(frames) >= max_frames):
            break
        frames.append(frame)
    cap.release()
    return frames


def frames_to_video(frames: list[np.ndarray], out_path: Path, fps: float = 25.0) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
