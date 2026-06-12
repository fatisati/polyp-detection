"""
One-shot training script — runs entirely in Modal cloud on A100.
Downloads Kvasir-SEG, converts masks → YOLO bboxes, fine-tunes YOLOv8s.
Saves best.pt to Modal Volume so inference/app.py can load it.

Run once:   modal run inference/train.py
Cost:       ~$0.70 (15-20 min on A100)
"""

import modal

app = modal.App("polyp-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["wget", "unzip", "libgl1", "libglib2.0-0"])
    .pip_install([
        "ultralytics>=8.3.0",
        "opencv-python-headless>=4.10.0",
        "numpy>=1.26.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
    ])
)

volume = modal.Volume.from_name("polyp-models", create_if_missing=True)
MODELS_DIR = "/models"
DATA_DIR = "/data"

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


@app.function(
    gpu="A100",
    image=image,
    volumes={MODELS_DIR: volume},
    timeout=3600,
)
def train():
    import os
    import shutil
    import subprocess
    import zipfile
    import cv2
    import numpy as np
    import yaml
    from pathlib import Path

    # ── 1. Download Kvasir-SEG ──────────────────────────────────────────────
    print("Downloading Kvasir-SEG...")
    subprocess.run(["wget", "-q", "-O", "/tmp/kvasir-seg.zip", KVASIR_URL], check=True)

    with zipfile.ZipFile("/tmp/kvasir-seg.zip", "r") as z:
        z.extractall("/tmp/kvasir-seg")

    # Kvasir-SEG unpacks to: kvasir-seg/images/*.jpg  kvasir-seg/masks/*.jpg
    src_images = Path("/tmp/kvasir-seg/kvasir-seg/images")
    src_masks  = Path("/tmp/kvasir-seg/kvasir-seg/masks")

    # ── 2. Convert masks → YOLO bbox labels ────────────────────────────────
    print("Converting segmentation masks to YOLO bounding boxes...")
    dataset = Path("/tmp/polyp-yolo")
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)

    image_files = sorted(src_images.glob("*.jpg"))
    split_idx = int(len(image_files) * 0.85)  # 85/15 train/val split

    for i, img_path in enumerate(image_files):
        split = "train" if i < split_idx else "val"
        mask_path = src_masks / img_path.name

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        h, w = mask.shape
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # One label line per connected polyp region
        label_lines = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < 100:   # skip tiny noise
                continue
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not label_lines:
            continue

        shutil.copy(img_path, dataset / "images" / split / img_path.name)
        (dataset / "labels" / split / img_path.stem).with_suffix(".txt").write_text(
            "\n".join(label_lines)
        )

    # ── 3. Write dataset YAML ───────────────────────────────────────────────
    dataset_yaml = dataset / "dataset.yaml"
    yaml.dump(
        {
            "path": str(dataset),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["polyp"],
        },
        open(dataset_yaml, "w"),
    )

    # ── 4. Train ────────────────────────────────────────────────────────────
    print("Training YOLOv8s on Kvasir-SEG (50 epochs)...")
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=32,
        device="cuda",
        project="/tmp/polyp-runs",
        name="kvasir",
        exist_ok=True,
        verbose=False,
    )

    # ── 5. Save best weights to Modal Volume ────────────────────────────────
    best_pt = Path("/tmp/polyp-runs/kvasir/weights/best.pt")
    dest = Path(MODELS_DIR) / "yolo_polyp_kvasir.pt"
    shutil.copy(best_pt, dest)
    volume.commit()

    print(f"\nTraining complete. Weights saved to Modal Volume: {dest}")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'see above'):.3f}")


@app.local_entrypoint()
def main():
    train.remote()
