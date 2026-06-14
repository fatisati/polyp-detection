"""
Concatenates frames from one colonoscopy sequence into a browser-compatible .mp4 (H.264).
Also exports a ground-truth JSON sidecar for overlay in the web UI.

Usage: python data/make_test_video.py [sequence_number]
Output:
  data/samples/videos/test_polyp_seq<N>.mp4
  data/samples/ground_truth/test_polyp_seq<N>_gt.json
"""
import cv2
import glob
import json
import os
import sys
import imageio

SEQ = sys.argv[1] if len(sys.argv) > 1 else "1"
base_dir = os.path.join(os.path.dirname(__file__), "TrainValid", "TrainValid")
frames_dir = os.path.join(base_dir, "Images", SEQ)
annots_dir = os.path.join(base_dir, "Annotations", SEQ)

samples_dir = os.path.join(os.path.dirname(__file__), "samples")
videos_dir  = os.path.join(samples_dir, "videos")
gt_dir      = os.path.join(samples_dir, "ground_truth")
os.makedirs(videos_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

out_video = os.path.join(videos_dir, f"test_polyp_seq{SEQ}.mp4")
out_gt    = os.path.join(gt_dir, f"test_polyp_seq{SEQ}_gt.json")

frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
if not frames:
    print(f"No frames found in {frames_dir}")
    sys.exit(1)

# Read frames as RGB
imgs = []
for f in frames:
    img = cv2.imread(f)
    if img is not None:
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

h, w = imgs[0].shape[:2]

# Write H.264 .mp4
writer = imageio.get_writer(out_video, fps=25, codec="libx264", quality=8)
for img in imgs:
    writer.append_data(img)
writer.close()
print(f"Video: {len(imgs)} frames -> {out_video}  ({w}x{h} @ 25fps, H.264)")

# Build ground-truth JSON from YOLO .txt annotations
gt_frames = []
for frame_path in frames:
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    txt_path = os.path.join(annots_dir, stem + ".txt")
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        # Format: first line = count, then lines of "x1 y1 x2 y2" in pixels
        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 4:
                x1, y1, x2, y2 = map(int, parts)
                boxes.append({"bbox": [x1, y1, x2, y2]})
    gt_frames.append(boxes)

with open(out_gt, "w") as f:
    json.dump({"width": w, "height": h, "frames": gt_frames}, f)

gt_count = sum(1 for f in gt_frames if f)
print(f"GT JSON: {gt_count}/{len(frames)} frames have annotations -> {out_gt}")
