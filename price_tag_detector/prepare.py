"""
prepare.py — Шаг 1
====================
Читает CSV-разметку, извлекает кадры из видео,
создаёт датасет в двух форматах:
  - YOLO format  →  dataset/images/ + dataset/labels/   (для YOLO11)
  - COCO JSON    →  dataset/annotations_*.json           (для DETR)

Дополнительно:
  - Умная аугментация (motion blur, low-light, perspective)
  - Визуальная проверка bbox (папка debug_vis/)
  - Подробные логи на каждом шаге

Запуск:
    python prepare.py
    python prepare.py --debug     # сохранить debug изображения с bbox
    python prepare.py --no-aug    # без аугментации
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ─── Настройки ────────────────────────────────────────────────────────────────
INPUT_DIR    = "input_videos"
DATASET_DIR  = "dataset"
ANNOT_DIR    = "annotations"
DEBUG_DIR    = "debug_vis"

VAL_SPLIT    = 0.2
MIN_BOX_PX   = 15       # минимальный размер bbox (пикс.)
RANDOM_SEED  = 42
AUG_FACTOR   = 3        # сколько аугментированных копий на каждый кадр
# ──────────────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# =============================================================================
# 1. Чтение CSV
# =============================================================================

def read_csv(csv_path):
    """
    Читает CSV разметку. Поддерживает запятую как разделитель дробной части.
    Возвращает список {timestamp_ms, bbox:[x1,y1,x2,y2]}.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Нормализуем имена столбцов
        raw_fields = reader.fieldnames or []
        norm_fields = [c.strip().lower().replace(" ", "_") for c in raw_fields]
        reader.fieldnames = norm_fields

        required = {"frame_timestamp", "x_min", "y_min", "x_max", "y_max"}
        missing  = required - set(norm_fields)
        if missing:
            print(f"    [ERR] CSV '{csv_path.name}': отсутствуют столбцы {missing}")
            print(f"           Найдены: {norm_fields}")
            return []

        skipped = 0
        for i, row in enumerate(reader, 1):
            try:
                ts   = int(float(str(row["frame_timestamp"]).replace(",", ".")))
                xmin = int(float(str(row["x_min"]).replace(",", ".")))
                ymin = int(float(str(row["y_min"]).replace(",", ".")))
                xmax = int(float(str(row["x_max"]).replace(",", ".")))
                ymax = int(float(str(row["y_max"]).replace(",", ".")))
            except (ValueError, KeyError) as e:
                print(f"    [skip] строка {i}: {e}")
                skipped += 1
                continue

            if xmax <= xmin or ymax <= ymin:
                print(f"    [skip] строка {i}: некорректный bbox"
                      f" [{xmin},{ymin},{xmax},{ymax}]")
                skipped += 1
                continue

            rows.append({"timestamp_ms": ts, "bbox": [xmin, ymin, xmax, ymax]})

    print(f"    CSV: считано {len(rows)} аннотаций, пропущено {skipped}")
    return rows


# =============================================================================
# 2. Извлечение кадров
# =============================================================================

def extract_frame(cap, timestamp_ms):
    """Извлекает кадр по временной метке в мс."""
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


# =============================================================================
# 3. Конвертация координат
# =============================================================================

def clip_bbox(xmin, ymin, xmax, ymax, fw, fh):
    """Обрезает bbox по границам кадра."""
    xmin = max(0, min(xmin, fw - 1))
    ymin = max(0, min(ymin, fh - 1))
    xmax = max(0, min(xmax, fw))
    ymax = max(0, min(ymax, fh))
    return xmin, ymin, xmax, ymax


def bbox_to_yolo(xmin, ymin, xmax, ymax, fw, fh):
    """Абсолютные пиксели → нормализованный YOLO формат (cx cy w h)."""
    xmin, ymin, xmax, ymax = clip_bbox(xmin, ymin, xmax, ymax, fw, fh)
    w_px = xmax - xmin
    h_px = ymax - ymin
    if w_px < MIN_BOX_PX or h_px < MIN_BOX_PX:
        return None
    cx = (xmin + xmax) / 2.0 / fw
    cy = (ymin + ymax) / 2.0 / fh
    w  = w_px / fw
    h  = h_px / fh
    return cx, cy, w, h


# =============================================================================
# 4. Аугментация (без внешних библиотек кроме OpenCV + numpy)
# =============================================================================

def apply_augmentation(frame, bboxes, aug_idx):
    """
    Применяет одну из аугментаций к кадру.
    bboxes: список [xmin, ymin, xmax, ymax] в абсолютных пикселях.
    Возвращает (aug_frame, aug_bboxes) или None если аугментация неприменима.
    """
    h, w = frame.shape[:2]
    aug_type = aug_idx % 6

    if aug_type == 0:
        # Motion blur (имитация движения камеры)
        size = random.choice([7, 11, 15])
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0 / size
        kernel = cv2.warpAffine(kernel, M, (size, size))
        kernel = kernel / (kernel.sum() + 1e-6)
        result = cv2.filter2D(frame, -1, kernel)
        return result, bboxes

    elif aug_type == 1:
        # Изменение яркости/контраста (разное освещение в магазине)
        alpha = random.uniform(0.5, 1.5)   # контраст
        beta  = random.uniform(-40, 40)    # яркость
        result = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return result, bboxes

    elif aug_type == 2:
        # Gaussian blur (расфокус)
        k = random.choice([3, 5, 7])
        result = cv2.GaussianBlur(frame, (k, k), 0)
        return result, bboxes

    elif aug_type == 3:
        # Горизонтальный флип
        result = cv2.flip(frame, 1)
        new_bboxes = []
        for (xmin, ymin, xmax, ymax) in bboxes:
            new_bboxes.append([w - xmax, ymin, w - xmin, ymax])
        return result, new_bboxes

    elif aug_type == 4:
        # JPEG артефакты (имитация сжатого видео)
        quality = random.randint(40, 75)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", frame, encode_param)
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return result, bboxes

    elif aug_type == 5:
        # Low-light + шум (плохое освещение)
        dark = (frame * random.uniform(0.3, 0.6)).astype(np.uint8)
        noise = np.random.normal(0, 15, dark.shape).astype(np.int16)
        result = np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return result, bboxes

    return frame, bboxes


# =============================================================================
# 5. Сохранение датасета
# =============================================================================

def save_sample(frame, bboxes, fw, fh, split, base_name, dataset_dir):
    """
    Сохраняет один кадр + YOLO label.
    Возвращает True если хотя бы один bbox валидный.
    """
    yolo_lines = []
    for (xmin, ymin, xmax, ymax) in bboxes:
        result = bbox_to_yolo(xmin, ymin, xmax, ymax, fw, fh)
        if result:
            cx, cy, bw, bh = result
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not yolo_lines:
        return False

    img_path = Path(dataset_dir, "images", split, f"{base_name}.jpg")
    lbl_path = Path(dataset_dir, "labels", split, f"{base_name}.txt")
    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
    return True


# =============================================================================
# 6. COCO JSON для DETR
# =============================================================================

class CocoBuilder:
    """Накапливает аннотации и сохраняет в COCO JSON формат."""
    def __init__(self):
        self.images = []
        self.annotations = []
        self._img_id  = 0
        self._ann_id  = 0

    def add(self, img_filename, fw, fh, bboxes_abs):
        self._img_id += 1
        self.images.append({
            "id": self._img_id,
            "file_name": img_filename,
            "width": fw,
            "height": fh,
        })
        for (xmin, ymin, xmax, ymax) in bboxes_abs:
            xmin, ymin, xmax, ymax = clip_bbox(xmin, ymin, xmax, ymax, fw, fh)
            if (xmax - xmin) < MIN_BOX_PX or (ymax - ymin) < MIN_BOX_PX:
                continue
            self._ann_id += 1
            self.annotations.append({
                "id": self._ann_id,
                "image_id": self._img_id,
                "category_id": 1,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO: x,y,w,h
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0,
            })

    def save(self, path):
        data = {
            "info": {"description": "Lenta Price Tag Dataset"},
            "categories": [{"id": 1, "name": "price_tag", "supercategory": "none"}],
            "images": self.images,
            "annotations": self.annotations,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    COCO JSON: {path}"
              f" ({len(self.images)} imgs, {len(self.annotations)} anns)")


# =============================================================================
# 7. Главная функция
# =============================================================================

def find_pairs(input_dir):
    pairs = []
    for f in sorted(Path(input_dir).iterdir()):
        if f.suffix.lower() in VIDEO_EXTS:
            csv_path = f.with_suffix(".csv")
            if csv_path.exists():
                pairs.append((f, csv_path))
                print(f"  [OK] {f.name} + {csv_path.name}")
            else:
                print(f"  [!]  Нет CSV для {f.name} (ожидался {csv_path.name})")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",   action="store_true",
                        help="Сохранить debug-изображения с нарисованными bbox")
    parser.add_argument("--no-aug",  action="store_true",
                        help="Отключить аугментацию")
    args = parser.parse_args()

    print("=" * 60)
    print("  ШАГ 1: CSV → YOLO + COCO датасет")
    print("=" * 60)

    # Создаём структуру папок
    for split in ("train", "val"):
        Path(DATASET_DIR, "images", split).mkdir(parents=True, exist_ok=True)
        Path(DATASET_DIR, "labels", split).mkdir(parents=True, exist_ok=True)
    Path(DATASET_DIR, "annotations").mkdir(parents=True, exist_ok=True)
    Path(ANNOT_DIR).mkdir(parents=True, exist_ok=True)
    if args.debug:
        Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(INPUT_DIR)
    if not pairs:
        print(f"\n[ERR] Нет пар video+CSV в '{INPUT_DIR}'")
        print("  Убедитесь что CSV называется так же как видео:")
        print("    input_videos/video_001.mp4")
        print("    input_videos/video_001.csv")
        return

    random.seed(RANDOM_SEED)

    coco_train = CocoBuilder()
    coco_val   = CocoBuilder()

    total_train = total_val = total_skip = total_aug = 0

    for video_path, csv_path in pairs:
        print(f"\n{'─'*55}")
        print(f"  Обработка: {video_path.name}")
        print(f"{'─'*55}")

        # 1. Читаем CSV
        annotations = read_csv(csv_path)
        if not annotations:
            print("  [!] Нет аннотаций, пропускаем")
            continue

        # 2. Сохраняем промежуточный JSON (для проверки)
        json_path = Path(ANNOT_DIR) / f"{video_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "video": video_path.name,
                "total": len(annotations),
                "annotations": annotations,
            }, f, ensure_ascii=False, indent=2)
        print(f"    JSON сохранён: {json_path}")

        # 3. Открываем видео
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"    [ERR] Не удалось открыть видео!")
            continue

        fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"    Разрешение: {fw}×{fh}, FPS: {fps:.1f}, кадров: {total}")

        # 4. Группируем аннотации по timestamp
        ts_map = {}
        for ann in annotations:
            ts_map.setdefault(ann["timestamp_ms"], []).append(ann["bbox"])

        print(f"    Уникальных timestamp: {len(ts_map)}")

        for ts_ms, bboxes in sorted(ts_map.items()):
            frame = extract_frame(cap, ts_ms)
            if frame is None:
                print(f"    [skip] кадр не найден @ {ts_ms}ms")
                total_skip += 1
                continue

            # Проверяем что хотя бы один bbox валидный
            valid_bboxes = [b for b in bboxes
                            if (b[2]-b[0]) >= MIN_BOX_PX and (b[3]-b[1]) >= MIN_BOX_PX]
            if not valid_bboxes:
                total_skip += 1
                continue

            split     = "val" if random.random() < VAL_SPLIT else "train"
            base_name = f"{video_path.stem}_{ts_ms}ms"

            # Сохраняем оригинальный кадр
            saved = save_sample(frame, valid_bboxes, fw, fh, split, base_name, DATASET_DIR)
            if not saved:
                total_skip += 1
                continue

            # COCO аннотация
            img_filename = f"{base_name}.jpg"
            if split == "train":
                coco_train.add(img_filename, fw, fh, valid_bboxes)
                total_train += 1
            else:
                coco_val.add(img_filename, fw, fh, valid_bboxes)
                total_val += 1

            # Debug визуализация
            if args.debug:
                dbg = frame.copy()
                for (x1, y1, x2, y2) in valid_bboxes:
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(dbg, "price_tag", (x1, max(y1-6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(str(Path(DEBUG_DIR) / f"{base_name}.jpg"), dbg,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Аугментация (только для train)
            if not args.no_aug and split == "train":
                for aug_i in range(AUG_FACTOR):
                    aug_frame, aug_bboxes = apply_augmentation(
                        frame.copy(), [b[:] for b in valid_bboxes], aug_i
                    )
                    aug_name = f"{base_name}_aug{aug_i}"
                    aug_saved = save_sample(
                        aug_frame, aug_bboxes, fw, fh, "train", aug_name, DATASET_DIR
                    )
                    if aug_saved:
                        coco_train.add(f"{aug_name}.jpg", fw, fh, aug_bboxes)
                        total_aug += 1

        cap.release()

    # 5. Сохраняем COCO JSON
    coco_train.save(Path(DATASET_DIR, "annotations", "instances_train.json"))
    coco_val.save(Path(DATASET_DIR, "annotations", "instances_val.json"))

    # 6. Создаём dataset.yaml для YOLO
    yaml_path = Path(DATASET_DIR) / "dataset.yaml"
    yaml_path.write_text(
        f"path: {Path(DATASET_DIR).resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: 1\n"
        f"names: [\"price_tag\"]\n",
        encoding="utf-8"
    )

    # Итоговая статистика
    train_imgs = len(list(Path(DATASET_DIR, "images", "train").glob("*.jpg")))
    val_imgs   = len(list(Path(DATASET_DIR, "images", "val").glob("*.jpg")))

    print("\n" + "=" * 60)
    print("  ГОТОВО!")
    print(f"  Train кадров (оригинал):  {total_train}")
    print(f"  Train кадров (с aug):     {total_train + total_aug}  (+{total_aug} aug)")
    print(f"  Val кадров:               {total_val}")
    print(f"  Пропущено:                {total_skip}")
    print(f"  Изображений в train/:     {train_imgs}")
    print(f"  Изображений в val/:       {val_imgs}")
    print(f"  dataset.yaml:             {yaml_path}")
    if args.debug:
        print(f"  Debug bbox визуализация:  {DEBUG_DIR}/")
    print("=" * 60)
    print("\nСледующий шаг:")
    print("  python train.py              ← YOLO11 (быстро, хорошее качество)")



if __name__ == "__main__":
    main()
