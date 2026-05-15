"""
Price Tag Detector & Tracker
Detects price tags using YOLOv8n, tracks them across frames,
selects the sharpest frame per track, saves crop + CSV.
"""

import cv2
import csv
import time
import uuid
import logging
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameSnapshot:
    """Single frame data for a tracked object."""
    timestamp_ms: float          # ms from video start
    bbox: Tuple[int, int, int, int]  # x, y, w, h (absolute pixels)
    sharpness: float
    frame: Optional[np.ndarray] = None  # stored only for best frame


@dataclass
class Track:
    """Lifecycle of one detected price tag."""
    track_id: int
    unique_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    best: Optional[FrameSnapshot] = None
    last_seen_frame: int = 0
    finished: bool = False

    def update(self, snapshot: FrameSnapshot, frame_idx: int):
        self.last_seen_frame = frame_idx
        if self.best is None or snapshot.sharpness > self.best.sharpness:
            self.best = snapshot

# ──────────────────────────────────────────────────────────────────────────────
# Sharpness metric
# ──────────────────────────────────────────────────────────────────────────────

def laplacian_variance(gray_crop: np.ndarray) -> float:
    """Higher = sharper. Fast and reliable focus measure."""
    if gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def tenengrad(gray_crop: np.ndarray) -> float:
    """Gradient energy – alternative sharpness metric."""
    gx = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def sharpness_score(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return laplacian_variance(gray)

# ──────────────────────────────────────────────────────────────────────────────
# IoU helper for simple tracker fallback
# ──────────────────────────────────────────────────────────────────────────────

def iou(b1, b2) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(x1, x2); iy = max(y1, y2)
    ix2 = min(x1+w1, x2+w2); iy2 = min(y1+h1, y2+h2)
    inter = max(0, ix2-ix) * max(0, iy2-iy)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Core processor
# ──────────────────────────────────────────────────────────────────────────────

class PriceTagProcessor:
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_missed_frames: int = 30,
        use_gpu: bool = True,
        min_crop_px: int = 20,
    ):
        self.output_dir = Path(output_dir)
        self.crops_dir = self.output_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.min_crop_px = min_crop_px

        device = "0" if use_gpu else "cpu"
        logging.info(f"Loading model: {model_path}  device={device}")
        self.model = YOLO(model_path)
        self.device = device

        self.tracks: Dict[int, Track] = {}   # tracker_id → Track
        self.finished_tracks: List[Track] = []

    # ── video processing ──────────────────────────────────────────────────────

    def process_video(self, video_path: str) -> List[dict]:
        """Process one video. Returns list of result dicts."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Video: {video_path}  fps={fps:.2f}  frames={total}")

        frame_idx = 0
        t0 = time.time()

        for result in self.model.track(
            source=video_path,
            stream=True,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            verbose=False,
        ):
            timestamp_ms = (frame_idx / fps) * 1000.0
            frame_bgr = result.orig_img

            if result.boxes is not None and result.boxes.id is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), tid in zip(boxes_xyxy, track_ids):
                    x, y = max(0, x1), max(0, y1)
                    w = max(1, x2 - x1)
                    h = max(1, y2 - y1)

                    crop = frame_bgr[y:y+h, x:x+w]
                    if crop.shape[0] < self.min_crop_px or crop.shape[1] < self.min_crop_px:
                        continue

                    score = sharpness_score(crop)
                    snap = FrameSnapshot(
                        timestamp_ms=timestamp_ms,
                        bbox=(x, y, w, h),
                        sharpness=score,
                        frame=crop.copy(),
                    )

                    if tid not in self.tracks:
                        self.tracks[tid] = Track(track_id=tid)
                    self.tracks[tid].update(snap, frame_idx)

            # Retire tracks not seen for a while
            for tid, track in list(self.tracks.items()):
                if frame_idx - track.last_seen_frame > self.max_missed_frames:
                    track.finished = True
                    self.finished_tracks.append(track)
                    del self.tracks[tid]

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                logging.info(f"  frame {frame_idx}/{total}  elapsed={elapsed:.1f}s")

        cap.release()

        # Retire all remaining tracks
        for track in self.tracks.values():
            track.finished = True
            self.finished_tracks.append(track)
        self.tracks.clear()

        # Save results
        video_name = Path(video_path).stem
        results = self._save_results(video_name)
        self.finished_tracks.clear()
        return results

    # ── save crops + return records ───────────────────────────────────────────

    def _save_results(self, video_name: str) -> List[dict]:
        records = []
        for track in self.finished_tracks:
            if track.best is None:
                continue
            snap = track.best
            x, y, w, h = snap.bbox
            ts = int(round(snap.timestamp_ms))

            # Filename: videoname__uniqueid__timestampms.jpg
            img_name = f"{video_name}__{track.unique_id}__{ts}ms.jpg"
            img_path = self.crops_dir / img_name
            cv2.imwrite(str(img_path), snap.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            records.append({
                "video": video_name,
                "unique_id": track.unique_id,
                "tracker_id": track.track_id,
                "timestamp_ms": ts,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "sharpness": round(snap.sharpness, 2),
                "crop_file": img_name,
            })

        logging.info(f"  Saved {len(records)} crops for '{video_name}'")
        return records


# ──────────────────────────────────────────────────────────────────────────────
# CSV writer
# ──────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "video", "unique_id", "tracker_id",
    "timestamp_ms", "x", "y", "w", "h",
    "sharpness", "crop_file",
]


def write_csv(records: List[dict], csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    logging.info(f"CSV saved: {csv_path}  ({len(records)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Price-tag detector & tracker")
    parser.add_argument(
        "--videos", nargs="+", required=True,
        help="Path(s) to input video file(s). Glob patterns supported.",
    )
    parser.add_argument(
        "--model", default="models/yolov8n.pt",
        help="Path to YOLO model weights (default: models/yolov8n.pt)",
    )
    parser.add_argument(
        "--output", default="output",
        help="Output directory for crops and CSV",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection confidence threshold (0..1)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--max-missed", type=int, default=30,
        help="Frames without detection before track is retired",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (disables CUDA)",
    )
    args = parser.parse_args()

    # Expand glob patterns
    from glob import glob
    video_paths = []
    for pattern in args.videos:
        matched = glob(pattern)
        if matched:
            video_paths.extend(matched)
        elif Path(pattern).exists():
            video_paths.append(pattern)
        else:
            logging.warning(f"No files matched: {pattern}")

    if not video_paths:
        logging.error("No video files found. Exiting.")
        return

    processor = PriceTagProcessor(
        model_path=args.model,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_missed_frames=args.max_missed,
        use_gpu=not args.cpu,
    )

    all_records = []
    for vp in video_paths:
        logging.info(f"=== Processing: {vp} ===")
        try:
            records = processor.process_video(vp)
            all_records.extend(records)
        except Exception as e:
            logging.error(f"Failed on {vp}: {e}", exc_info=True)

    csv_path = str(Path(args.output) / "results.csv")
    write_csv(all_records, csv_path)
    logging.info(f"Done. Total price tags saved: {len(all_records)}")


if __name__ == "__main__":
    main()
