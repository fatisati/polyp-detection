"""Thin async wrapper around the Modal PolypDetector class."""

import asyncio
import time

import modal

# Cached once per backend process — avoids a Modal API lookup on every call
_detector_handle = None

def _detector():
    global _detector_handle
    if _detector_handle is None:
        _detector_handle = modal.Cls.from_name("polyp-detection", "PolypDetector")()
    return _detector_handle


async def warmup() -> str:
    return await asyncio.to_thread(_detector().warmup.remote)


async def infer_frame(frame_bytes: bytes) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    result = await _detector().infer_frame.remote.aio(frame_bytes)
    modal_ms = int((time.perf_counter() - t0) * 1000)
    return result, {"modal_ms": modal_ms}


async def infer_video(video_bytes: bytes) -> dict:
    return await asyncio.to_thread(_detector().infer_video.remote, video_bytes)
