"""
infer.py — Шаг 3
==================
Два режима inference в одном файле:

  python infer.py                       → YOLO11 batch (все видео из input_videos/)
  python infer.py --source video.mp4    → одно видео
  python infer.py --source 0 --preview  → webcam
  python infer.py --conf 0.2            → изменить порог
  python infer.py --eval                → считать mAP на val наборе после inference

Для обоих режимов:
  - IoU-трекер (для DETR) / ByteTracker (для YOLO11)
  - Выбор лучшего кадра: резкость + confidence + площадь + угол
  - output/crops/   — вырезанные ценники
  - output/results.csv — координаты и метаданные
"""

import argparse
import csv
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

# ─── Настройки ────────────────────────────────────────────────────────────────
YOLO_MODEL     = "runs/train/yolo11/weights/best.pt"
DETR_MODEL_DIR = "runs/train/detr"
INPUT_DIR      = "input_videos"
OUTPUT_DIR     = "output"
CONF_THRESHOLD = 0.25     # YOLO: 0.25 достаточно; DETR: 0.4-0.5
IOU_THRESHOLD  = 0.45
MAX_MISSED     = 30
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_YOLO    = "0" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
CSV_FIELDS = [
    "filename", "track_id", "unique_id",
    "frame_timestamp_ms", "x_min", "y_min", "x_max", "y_max",
    "confidence", "sharpness", "quality_score", "crop_file",
]


# =============================================================================
# Утилиты: резкость и scoring
# =============================================================================

def calc_sharpness(crop):
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calc_quality(crop, conf, x1, y1, x2, y2, fw, fh):
    """
    Итоговый скор кадра:
      40% резкость (Лапласиан)
    + 30% confidence модели
    + 20% относительная площадь bbox
    + 10% соотношение сторон (ценники обычно широкие)
    """
    sharp = min(calc_sharpness(crop) / 500.0, 1.0)
    area  = min(((x2 - x1) * (y2 - y1)) / max(fw * fh, 1), 1.0)
    ratio = max(x2 - x1, 1) / max(y2 - y1, 1)
    angle = (1.0 if 0.8 <= ratio <= 4.0
             else (ratio / 0.8 if ratio < 0.8 else 4.0 / ratio))
    return 0.40 * sharp + 0.30 * min(conf, 1.0) + 0.20 * area + 0.10 * angle


def calc_iou(b1, b2):
    ix  = max(b1[0], b2[0]); iy  = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = ((b1[2]-b1[0])*(b1[3]-b1[1]) +
             (b2[2]-b2[0])*(b2[3]-b2[1]) - inter)
    return inter / max(union, 1)


# =============================================================================
# Трекер (общий для обоих режимов)
# =============================================================================

class Track:
    def __init__(self, tid):
        self.track_id   = tid
        self.unique_id  = str(uuid.uuid4())[:8]
        self.best_score = -1.0
        self.best_snap  = None   # (ts_ms, bbox, conf, crop, score)
        self.last_frame = 0
        self.last_bbox  = None

    def update(self, ts_ms, bbox, conf, crop, frame_idx, fw, fh):
        x1, y1, x2, y2 = bbox
        score = calc_quality(crop, conf, x1, y1, x2, y2, fw, fh)
        if score > self.best_score:
            self.best_score = score
            self.best_snap  = (ts_ms, bbox, conf, crop.copy(), score)
        self.last_frame = frame_idx
        self.last_bbox  = bbox


class IoUTracker:
    """
    Простой IoU-трекер.
    Используется для DETR (ByteTracker встроен только в ultralytics).
    """
    def __init__(self, max_missed=30, iou_thresh=0.4):
        self.max_missed = max_missed
        self.iou_thresh = iou_thresh
        self.active     = {}
        self.finished   = []
        self._next_id   = 0

    def update(self, detections, frame_idx, ts_ms, fw, fh):
        unmatched = []
        used_tids = set()

        for det in detections:
            best_iou = 0.0
            best_tid = None
            for tid, track in self.active.items():
                if tid in used_tids:
                    continue
                if track.last_bbox is None:
                    continue
                v = calc_iou(det["bbox"], track.last_bbox)
                if v > best_iou:
                    best_iou = v
                    best_tid = tid

            if best_iou >= self.iou_thresh and best_tid is not None:
                self.active[best_tid].update(
                    ts_ms, det["bbox"], det["conf"], det["crop"],
                    frame_idx, fw, fh)
                used_tids.add(best_tid)
            else:
                unmatched.append(det)

        for det in unmatched:
            tid = self._next_id
            self._next_id += 1
            t = Track(tid)
            t.update(ts_ms, det["bbox"], det["conf"], det["crop"],
                     frame_idx, fw, fh)
            self.active[tid] = t

        done = [tid for tid, t in self.active.items()
                if frame_idx - t.last_frame > self.max_missed]
        for tid in done:
            self.finished.append(self.active.pop(tid))

    def pop_finished(self):
        result = [t for t in self.finished if t.best_snap is not None]
        self.finished.clear()
        return result

    def flush(self):
        for t in self.active.values():
            self.finished.append(t)
        self.active.clear()
        result = [t for t in self.finished if t.best_snap is not None]
        self.finished.clear()
        return result


# =============================================================================
# Сохранение результата
# =============================================================================

def save_track(track, video_path, crops_dir):
    if track.best_snap is None:
        return None
    ts_ms, bbox, conf, crop, score = track.best_snap
    x1, y1, x2, y2 = bbox
    crop_name = f"{video_path.stem}__{track.unique_id}__{int(ts_ms)}ms.jpg"
    cv2.imwrite(str(Path(crops_dir) / crop_name), crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {
        "filename":           video_path.name,
        "track_id":           track.track_id,
        "unique_id":          track.unique_id,
        "frame_timestamp_ms": int(round(ts_ms)),
        "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
        "confidence":         round(conf, 4),
        "sharpness":          round(calc_sharpness(crop), 2),
        "quality_score":      round(score, 4),
        "crop_file":          crop_name,
    }


def flush_tracks(tracker, video_path, crops_dir, records):
    for track in tracker.flush():
        rec = save_track(track, video_path, crops_dir)
        if rec:
            records.append(rec)


def process_finished(tracker, video_path, crops_dir, records):
    for track in tracker.pop_finished():
        rec = save_track(track, video_path, crops_dir)
        if rec:
            records.append(rec)
            print(f"    [+] ID:{track.track_id}  "
                  f"score={track.best_score:.3f}  "
                  f"t={int(track.best_snap[0])}ms")


# =============================================================================
# YOLO11 inference
# =============================================================================

def process_video_yolo(video_path, model, crops_dir, show_preview):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERR] Не удалось открыть: {video_path}")
        return []

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  {fw}×{fh} @ {fps:.1f}fps  кадров:{total}")

    # Для YOLO используем встроенный ByteTracker через model.track()
    tracker   = IoUTracker(max_missed=MAX_MISSED)
    records   = []
    frame_idx = 0

    for result in model.track(
        source=str(video_path),
        stream=True,
        device=DEVICE_YOLO,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        persist=True,
        verbose=False,
        agnostic_nms=True,
    ):
        ts_ms     = frame_idx / fps * 1000.0
        frame_bgr = result.orig_img
        detections = []

        if result.boxes is not None and result.boxes.id is not None:
            for box, tid, conf in zip(
                result.boxes.xyxy.cpu().numpy().astype(int),
                result.boxes.id.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
            ):
                x1 = max(0, box[0]); y1 = max(0, box[1])
                x2 = min(fw, box[2]); y2 = min(fh, box[3])
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                detections.append({
                    "track_id": int(tid),
                    "bbox":     (x1, y1, x2, y2),
                    "conf":     float(conf),
                    "crop":     crop,
                })
                if show_preview:
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(frame_bgr,
                                f"ID:{tid} {conf:.2f}",
                                (x1, max(y1 - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        tracker.update(detections, frame_idx, ts_ms, fw, fh)
        process_finished(tracker, video_path, crops_dir, records)

        if show_preview:
            cv2.putText(frame_bgr, f"YOLO11  conf:{CONF_THRESHOLD}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.imshow(f"YOLO11 — {video_path.name}  [Q=quit]", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 150 == 0:
            print(f"  {frame_idx}/{total} кадров...")

    flush_tracks(tracker, video_path, crops_dir, records)
    if show_preview:
        cv2.destroyAllWindows()
    return records


# =============================================================================
# DINOv2+DETR inference
# =============================================================================

def load_detr_model(model_dir):
    from transformers import AutoImageProcessor, DetrForObjectDetection
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    weights_path = Path(model_dir) / "best.pt"
    state = torch.load(str(weights_path), map_location=DEVICE)
    # Загружаем только совпадающие ключи (backbone мог измениться)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"  Загружено весов: {len(filtered)}/{len(model_state)}")
    return model, processor


def detect_frame_detr(model, processor, frame_bgr, conf_thresh, fw, fh):
    from PIL import Image
    img_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs  = processor(images=img_rgb, return_tensors="pt")
    inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[fh, fw]], device=DEVICE)
    results = processor.post_process_object_detection(
        outputs, threshold=conf_thresh, target_sizes=target_sizes
    )[0]

    detections = []
    for score, box in zip(results["scores"].cpu(), results["boxes"].cpu()):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw, x2); y2 = min(fh, y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "conf": float(score),
            "crop": crop,
        })
    return detections


def process_video_detr(video_path, model, processor, crops_dir, show_preview):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERR] {video_path}"); return []

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {fw}×{fh} @ {fps:.1f}fps  кадров:{total}")

    tracker   = IoUTracker(max_missed=MAX_MISSED)
    records   = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        ts_ms      = frame_idx / fps * 1000.0
        detections = detect_frame_detr(model, processor, frame_bgr,
                                       CONF_THRESHOLD, fw, fh)

        tracker.update(detections, frame_idx, ts_ms, fw, fh)
        process_finished(tracker, video_path, crops_dir, records)

        if show_preview:
            disp = frame_bgr.copy()
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(disp, f"{d['conf']:.2f}",
                            (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
            cv2.putText(disp, f"DETR  conf:{CONF_THRESHOLD}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.imshow(f"DETR — {video_path.name}  [Q=quit]", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total} кадров...")

    cap.release()
    flush_tracks(tracker, video_path, crops_dir, records)
    if show_preview:
        cv2.destroyAllWindows()
    return records


# =============================================================================
# mAP eval (через ultralytics val для YOLO)
# =============================================================================

def run_eval_yolo(model_path, dataset_yaml):
    from ultralytics import YOLO
    print("\n  Запуск val evaluation...")
    model   = YOLO(model_path)
    metrics = model.val(data=dataset_yaml, device=DEVICE_YOLO, verbose=False)
    print()
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    global CONF_THRESHOLD

    parser = argparse.ArgumentParser(description="Lenta Price Tag — Inference")
    parser.add_argument("--mode",    default="yolo",
                        choices=["yolo", "detr"],
                        help="yolo = YOLO11, detr = DINOv2+DETR")
    parser.add_argument("--source",  default=None,
                        help="Путь к видео / '0' для webcam (по умолчанию — input_videos/)")
    parser.add_argument("--model",   default=None,
                        help="Путь к весам (переопределяет дефолт)")
    parser.add_argument("--output",  default=OUTPUT_DIR)
    parser.add_argument("--conf",    default=CONF_THRESHOLD, type=float)
    parser.add_argument("--preview", action="store_true",
                        help="Показывать окно с детекциями")
    parser.add_argument("--eval",    action="store_true",
                        help="Запустить eval на val датасете (только YOLO)")
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf

    print("=" * 60)
    print(f"  ШАГ 3: Inference — {args.mode.upper()}")
    print("=" * 60)
    print(f"  Conf threshold: {CONF_THRESHOLD}")
    print(f"  Device: {DEVICE}")
    print()

    crops_dir = Path(args.output) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    # ── YOLO11 ────────────────────────────────────────────────────────────────
    if args.mode == "yolo":
        model_path = args.model or YOLO_MODEL
        if not Path(model_path).exists():
            print(f"[ERR] Модель не найдена: {model_path}")
            print("Сначала: python train.py")
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERR] pip install ultralytics>=8.3.0"); return

        model = YOLO(model_path)
        print(f"  Модель: {model_path}\n")

        # Eval mode
        if args.eval:
            yaml_path = Path(DATASET_DIR if "DATASET_DIR" in dir() else "dataset") / "dataset.yaml"
            if yaml_path.exists():
                run_eval_yolo(model_path, str(yaml_path))
            return

        if args.source:
            src = int(args.source) if args.source.isdigit() else args.source
            vp  = Path(args.source) if not str(args.source).isdigit() else Path("webcam")
            records = process_video_yolo(vp, model, crops_dir, args.preview)
            all_records.extend(records)
        else:
            videos = [f for f in Path(INPUT_DIR).iterdir()
                      if f.suffix.lower() in VIDEO_EXTS]
            if not videos:
                print(f"[ERR] Нет видео в '{INPUT_DIR}'"); return
            for vp in sorted(videos):
                print(f"\n  --- {vp.name} ---")
                recs = process_video_yolo(vp, model, crops_dir, args.preview)
                all_records.extend(recs)
                print(f"  Ценников: {len(recs)}")

    # ── DINOv2+DETR ───────────────────────────────────────────────────────────
    else:
        model_dir = args.model or DETR_MODEL_DIR
        weights   = Path(model_dir) / "best.pt"
        if not weights.exists():
            print(f"[ERR] Веса не найдены: {weights}")
            print("Сначала: python train.py --mode detr"); return

        try:
            model, processor = load_detr_model(model_dir)
        except Exception as e:
            print(f"[ERR] Ошибка загрузки: {e}"); return

        print(f"  Модель: {model_dir}\n")

        if args.source:
            vp = Path(args.source)
            all_records.extend(
                process_video_detr(vp, model, processor, crops_dir, args.preview))
        else:
            videos = [f for f in Path(INPUT_DIR).iterdir()
                      if f.suffix.lower() in VIDEO_EXTS]
            if not videos:
                print(f"[ERR] Нет видео в '{INPUT_DIR}'"); return
            for vp in sorted(videos):
                print(f"\n  --- {vp.name} ---")
                recs = process_video_detr(vp, model, processor, crops_dir, args.preview)
                all_records.extend(recs)
                print(f"  Ценников: {len(recs)}")

    # ── Сохранение CSV ────────────────────────────────────────────────────────
    csv_path = Path(args.output) / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 60)
    print(f"  ГОТОВО!")
    print(f"  Ценников найдено: {len(all_records)}")
    print(f"  CSV:   {csv_path}")
    print(f"  Crops: {crops_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
