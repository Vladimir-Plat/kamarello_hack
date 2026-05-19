"""
train.py — Шаг 2
==================
Два режима обучения в одном файле:

  python train.py              →  YOLO11n (быстро, хорошее качество)
  python train.py --mode detr  →  DINOv2 + DETR (медленнее, точнее)

YOLO11:
  - Transfer learning с COCO
  - ~5-15 мин на 50 эпох (RTX 4060)
  - mAP@0.5 обычно 0.85+ при достаточном датасете

DINOv2 + DETR:
  - DINOv2-small как backbone (замена ResNet в DETR)
  - End-to-end обучение без NMS
  - Лучше на сложных случаях (окклюзии, разные масштабы)
  - ~30-60 мин на 30 эпох
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DetrForObjectDetection, DetrConfig, AutoImageProcessor
from transformers.models.detr.modeling_detr import DetrDecoderOutput


# ─── Настройки YOLO11 ─────────────────────────────────────────────────────────
YOLO_MODEL = "yolo11n.pt"     # n / s / m / l / x
YOLO_EPOCHS = 100
YOLO_BATCH = 16               # уменьшить до 8 при OOM
YOLO_IMGSZ = 640
YOLO_PATIENCE = 20

# ─── Настройки DETR ───────────────────────────────────────────────────────────
DETR_EPOCHS = 50
DETR_BATCH = 4                # уменьшить до 2 при OOM
DETR_LR = 1e-4
DETR_LR_BACKBONE = 1e-5       # backbone обучается с меньшим LR
DETR_IMG_SIZE = 518           # DINOv2 требует фиксированный размер 518x518

# ─── Общие ────────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
MODEL_OUT = "runs/train"
DEVICE_YOLO = "0"             # "0"=GPU, "cpu"=CPU
DEVICE_TORCH = "cuda" if torch.cuda.is_available() else "cpu"

# Важно:
# Для HuggingFace DETR num_labels — количество foreground-классов.
# При одном классе должно быть 1.
# В COCO-аннотациях category_id при этом может быть 1.
NUM_CLASSES = 1
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# YOLO11
# =============================================================================

def train_yolo():
    print("=" * 60)
    print("  Режим: YOLO11")
    print("=" * 60)

    yaml_path = Path(DATASET_DIR) / "dataset.yaml"
    if not yaml_path.exists():
        print(f"[ERR] {yaml_path} не найден. Запустите: python prepare.py")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERR] pip install ultralytics>=8.3.0")
        return

    print(f"\n  Модель:   {YOLO_MODEL}")
    print(f"  Эпох:     {YOLO_EPOCHS}")
    print(f"  Батч:     {YOLO_BATCH}")
    print(f"  Размер:   {YOLO_IMGSZ}")
    print(f"  Device:   {DEVICE_YOLO}")
    print(f"  Patience: {YOLO_PATIENCE}")
    print()

    model = YOLO(YOLO_MODEL)

    results = model.train(
        data=str(yaml_path),
        epochs=YOLO_EPOCHS,
        batch=YOLO_BATCH,
        imgsz=YOLO_IMGSZ,
        device=DEVICE_YOLO,
        patience=YOLO_PATIENCE,
        project=MODEL_OUT,
        name="yolo11",
        # Transfer learning
        pretrained=True,
        # Mixed precision
        amp=True,
        # Оптимизатор
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Аугментация встроенная, дополняет offline-aug
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0003,
        flipud=0.0,
        fliplr=0.3,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Порог — recall важнее precision
        iou=0.7,
        conf=None,
        # Сохранение
        save_period=10,
        plots=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    last = Path(results.save_dir) / "weights" / "last.pt"

    print("\n" + "=" * 60)
    print("  YOLO11 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    if best.exists():
        print(f"  Лучшие веса: {best}")
    if last.exists():
        print(f"  Последние:   {last}")

    try:
        m = results.results_dict
        print()
        print(f"  mAP@0.5:      {m.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@0.5:0.95: {m.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision:    {m.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall:       {m.get('metrics/recall(B)', 0):.4f}")
    except Exception:
        pass

    print("=" * 60)
    print(f"\nДля inference: python infer.py --model {best}")


# =============================================================================
# DINOv2 + DETR
# =============================================================================

class CocoDetectionDataset(Dataset):
    """Читает COCO JSON аннотации для обучения DETR."""

    def __init__(self, images_dir, ann_file, processor):
        self.images_dir = Path(images_dir)
        self.processor = processor

        with open(ann_file, encoding="utf-8") as f:
            coco = json.load(f)

        # Индекс: image_id → список аннотаций
        self.ann_map = {}
        for ann in coco.get("annotations", []):
            self.ann_map.setdefault(ann["image_id"], []).append(ann)

        self.images = [
            img for img in coco.get("images", [])
            if (self.images_dir / img["file_name"]).exists()
        ]

        print(f"    Dataset: {len(self.images)} изображений из {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]

        image_path = self.images_dir / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        anns = self.ann_map.get(img_id, [])

        # COCO bbox: [x, y, w, h]
        coco_anns = []
        for ann in anns:
            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue

            x, y, w, h = bbox

            # Пропускаем битые bbox
            if w <= 0 or h <= 0:
                continue

            coco_anns.append(
                {
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "category_id": int(ann.get("category_id", 1)),
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )

        encoding = self.processor(
            images=image,
            annotations={
                "image_id": int(img_id),
                "annotations": coco_anns,
            },
            return_tensors="pt",
        )

        # ВАЖНО:
        # pixel_values / pixel_mask — tensors с batch-dim [1, ...], их squeeze можно.
        # labels — list[dict], его squeeze делать нельзя.
        item = {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }

        if "pixel_mask" in encoding:
            item["pixel_mask"] = encoding["pixel_mask"].squeeze(0)

        return item


def detr_collate(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    result = {
        "pixel_values": pixel_values,
        "labels": [item["labels"] for item in batch],
    }

    if "pixel_mask" in batch[0]:
        result["pixel_mask"] = torch.stack([item["pixel_mask"] for item in batch])

    return result


def build_dino_detr(num_classes, img_size=518):
    """
    Строит DETR с DINOv2-small backbone вместо ResNet.
    DINOv2 требует фиксированный размер 518x518.
    """
    config = DetrConfig(
        num_labels=num_classes,
        id2label={0: "price_tag"},
        label2id={"price_tag": 0},
        num_queries=50,
    )

    model = DetrForObjectDetection(config)

    try:
        import timm

        dinov2 = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            global_pool="",
            img_size=img_size,  # Важно: фиксированный размер для DINOv2
        )

        class DINOv2Backbone(nn.Module):
            """Адаптер DINOv2 → формат backbone для DETR."""
            
            def __init__(self, dinov2_model, out_channels=256):
                super().__init__()
                self.dinov2 = dinov2_model
                self.embed_dim = dinov2_model.embed_dim
                self.patch_size = dinov2_model.patch_embed.patch_size[0]
                self.img_size = img_size
                
                # Вычисляем размер feature map после patch embedding
                self.num_patches = (img_size // self.patch_size) ** 2
                self.grid_size = img_size // self.patch_size
                
                # Проекция из embedding dimension в out_channels
                self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)
                self.out_channels = [out_channels]
                
                # Для совместимости с DETR
                self.num_channels = out_channels
                self.channels = out_channels
            
            def forward(self, pixel_values, pixel_mask=None):
                # pixel_values: [B, C, H, W]
                B, C, H, W = pixel_values.shape
                
                # Принудительно изменяем размер до ожидаемого DINOv2
                if H != self.img_size or W != self.img_size:
                    pixel_values = torch.nn.functional.interpolate(
                        pixel_values, 
                        size=(self.img_size, self.img_size),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Forward через DINOv2
                x = self.dinov2.forward_features(pixel_values)
                
                # Извлекаем patch tokens (без cls token)
                if isinstance(x, dict):
                    # Для современных версий timm
                    if 'x_norm_patchtokens' in x:
                        patch_tokens = x['x_norm_patchtokens']
                    elif 'x_norm' in x:
                        patch_tokens = x['x_norm']
                        if patch_tokens.dim() == 3 and patch_tokens.shape[1] == self.num_patches + 1:
                            patch_tokens = patch_tokens[:, 1:, :]
                    else:
                        # Пробуем найти первый тензор в словаре
                        first_key = list(x.keys())[0]
                        patch_tokens = x[first_key]
                        if patch_tokens.dim() == 3 and patch_tokens.shape[1] == self.num_patches + 1:
                            patch_tokens = patch_tokens[:, 1:, :]
                else:
                    # x - тензор [B, N, D]
                    if x.shape[1] == self.num_patches + 1:
                        x = x[:, 1:, :]  # Убираем cls token
                    patch_tokens = x
                
                # Reshape обратно в 2D
                patch_tokens = patch_tokens.permute(0, 2, 1)
                patch_tokens = patch_tokens.reshape(
                    B, self.embed_dim, self.grid_size, self.grid_size
                )
                
                # Проекция в нужное число каналов
                feat = self.proj(patch_tokens)
                
                # Создаем маску если ее нет
                if pixel_mask is not None:
                    # Изменяем размер маски под размер feature map
                    if pixel_mask.shape[1] != self.grid_size or pixel_mask.shape[2] != self.grid_size:
                        pixel_mask = torch.nn.functional.interpolate(
                            pixel_mask.float().unsqueeze(1),
                            size=(self.grid_size, self.grid_size),
                            mode='nearest'
                        ).squeeze(1).bool()
                else:
                    # Создаем маску из единиц
                    pixel_mask = torch.ones(B, self.grid_size, self.grid_size, device=pixel_values.device, dtype=torch.bool)
                
                # Возвращаем объект DetrBackboneOutput, который ожидает DETR
                return DetrBackboneOutput(
                    feature_maps=[feat],
                    mask=pixel_mask,
                )
        
        backbone = DINOv2Backbone(dinov2, out_channels=256)
        
        # Полностью заменяем backbone модель
        model.model.backbone = backbone
        
        # Обновляем конфигурацию
        model.config.backbone_out_channels = 256
        model.config.hidden_size = 256
        model.config.d_model = 256
        
        # Создаем новый input_projection если нужно
        if hasattr(model.model, 'input_projection'):
            model.model.input_projection = nn.Conv2d(256, 256, kernel_size=1)
        
        print(f"    [OK] DINOv2-small backbone подключён (размер {img_size}x{img_size})")
        print(f"    [OK] Feature map размер: {img_size//14}x{img_size//14}")

    except Exception as e:
        print(f"    [!] DINOv2 недоступен или не подключился ({e})")
        print("    [!] Используем стандартный DETR-ResNet50 fallback")
        import traceback
        traceback.print_exc()

        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # Переназначаем маппинг классов
        model.config.id2label = {0: "price_tag"}
        model.config.label2id = {"price_tag": 0}

        print("    [OK] DETR-ResNet50 pretrained загружен как fallback")

    return model


def move_labels_to_device(labels, device):
    moved = []
    for label in labels:
        moved_label = {}
        for key, value in label.items():
            if isinstance(value, torch.Tensor):
                moved_label[key] = value.to(device)
            else:
                moved_label[key] = value
        moved.append(moved_label)
    return moved


def train_detr():
    print("=" * 60)
    print("  Режим: DINOv2 + DETR")
    print("=" * 60)
    print(f"\n  Device: {DEVICE_TORCH}")

    ann_train = Path(DATASET_DIR, "annotations", "instances_train.json")
    ann_val = Path(DATASET_DIR, "annotations", "instances_val.json")

    if not ann_train.exists():
        print(f"[ERR] {ann_train} не найден. Запустите: python prepare.py")
        return

    if not ann_val.exists():
        print(f"[ERR] {ann_val} не найден. Запустите: python prepare.py")
        return

    try:
        from transformers import AutoImageProcessor
    except ImportError:
        print("[ERR] pip install transformers>=4.40.0 timm")
        return

    print("\n  Загрузка processor и модели...")

    # Используем размер 518 для DINOv2
    processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": DETR_IMG_SIZE, "longest_edge": DETR_IMG_SIZE},
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
    )

    train_ds = CocoDetectionDataset(
        Path(DATASET_DIR, "images", "train"),
        ann_train,
        processor,
    )

    val_ds = CocoDetectionDataset(
        Path(DATASET_DIR, "images", "val"),
        ann_val,
        processor,
    )

    if len(train_ds) == 0:
        print("[ERR] Пустой train датасет")
        return

    if len(val_ds) == 0:
        print("[ERR] Пустой val датасет")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=DETR_BATCH,
        shuffle=True,
        collate_fn=detr_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=DETR_BATCH,
        shuffle=False,
        collate_fn=detr_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_dino_detr(NUM_CLASSES, img_size=DETR_IMG_SIZE)
    model.to(DEVICE_TORCH)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Параметров всего:    {total_params:,}")
    print(f"  Обучаемых:           {train_params:,}")

    backbone_params = [
        p for name, p in model.named_parameters()
        if "backbone" in name and p.requires_grad
    ]

    other_params = [
        p for name, p in model.named_parameters()
        if "backbone" not in name and p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": DETR_LR_BACKBONE},
            {"params": other_params, "lr": DETR_LR},
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=DETR_EPOCHS,
        eta_min=1e-6,
    )

    out_dir = Path(MODEL_OUT) / "detr"
    out_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(out_dir))

    best_val_loss = float("inf")
    history = []

    print(f"\n  Эпох: {DETR_EPOCHS}  Батч: {DETR_BATCH}  LR: {DETR_LR}\n")

    for epoch in range(1, DETR_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_losses = {"total": 0.0, "ce": 0.0, "bbox": 0.0, "giou": 0.0}
        n_train = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(DEVICE_TORCH)
            pixel_mask = batch.get("pixel_mask")

            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(DEVICE_TORCH)

            labels = move_labels_to_device(batch["labels"], DEVICE_TORCH)

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            train_losses["total"] += loss.item()

            ld = outputs.loss_dict if hasattr(outputs, "loss_dict") else {}
            train_losses["ce"] += ld.get("loss_ce", torch.tensor(0.0)).item()
            train_losses["bbox"] += ld.get("loss_bbox", torch.tensor(0.0)).item()
            train_losses["giou"] += ld.get("loss_giou", torch.tensor(0.0)).item()

            n_train += 1

        for key in train_losses:
            train_losses[key] /= max(n_train, 1)

        # ── Val ────────────────────────────────────────────────────────────────
        model.eval()
        val_losses = {"total": 0.0, "ce": 0.0, "bbox": 0.0, "giou": 0.0}
        n_val = 0
        n_detected = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE_TORCH)
                pixel_mask = batch.get("pixel_mask")

                if pixel_mask is not None:
                    pixel_mask = pixel_mask.to(DEVICE_TORCH)

                labels = move_labels_to_device(batch["labels"], DEVICE_TORCH)

                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )

                val_losses["total"] += outputs.loss.item()

                ld = outputs.loss_dict if hasattr(outputs, "loss_dict") else {}
                val_losses["ce"] += ld.get("loss_ce", torch.tensor(0.0)).item()
                val_losses["bbox"] += ld.get("loss_bbox", torch.tensor(0.0)).item()
                val_losses["giou"] += ld.get("loss_giou", torch.tensor(0.0)).item()

                n_val += 1

                # Proxy-метрика: количество предсказаний с confidence > 0.5
                probs = torch.softmax(outputs.logits, dim=-1)[..., :-1].max(-1).values
                n_detected += (probs > 0.5).sum().item()

        for key in val_losses:
            val_losses[key] /= max(n_val, 1)

        scheduler.step()

        ep_info = {
            "epoch": epoch,
            "train_loss": train_losses["total"],
            "train_ce": train_losses["ce"],
            "train_bbox": train_losses["bbox"],
            "train_giou": train_losses["giou"],
            "val_loss": val_losses["total"],
            "val_ce": val_losses["ce"],
            "val_bbox": val_losses["bbox"],
            "val_giou": val_losses["giou"],
            "detected_conf_gt_05": n_detected,
        }

        history.append(ep_info)

        print(
            f"  Epoch [{epoch:3d}/{DETR_EPOCHS}]  "
            f"train={train_losses['total']:.4f}  "
            f"val={val_losses['total']:.4f}  "
            f"(ce={val_losses['ce']:.3f} "
            f"bbox={val_losses['bbox']:.3f} "
            f"giou={val_losses['giou']:.3f})  "
            f"det>0.5:{n_detected}"
        )

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(model.state_dict(), str(out_dir / "best.pt"))
            print(f"    → best.pt сохранён (val_loss={best_val_loss:.4f})")

    torch.save(model.state_dict(), str(out_dir / "last.pt"))

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  DINOv2+DETR ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"  Лучший val_loss:  {best_val_loss:.4f}")
    print(f"  Веса:             {out_dir}/best.pt")
    print()
    print("  mAP@0.5:      (запустите: python infer.py --mode detr --eval)")
    print("  mAP@0.5:0.95: (запустите: python infer.py --mode detr --eval)")
    print("  Precision:    (запустите: python infer.py --mode detr --eval)")
    print("  Recall:       (запустите: python infer.py --mode detr --eval)")

    if history:
        print(f"  Val total loss: {best_val_loss:.4f}")
        print(f"  Val CE loss:    {history[-1]['val_ce']:.4f}")
        print(f"  Val BBox loss:  {history[-1]['val_bbox']:.4f}")
        print(f"  Val GIoU loss:  {history[-1]['val_giou']:.4f}")

    print("=" * 60)
    print("\nДля inference: python infer.py --mode detr")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Lenta Price Tag — Train")

    parser.add_argument(
        "--mode",
        default="yolo",
        choices=["yolo", "detr"],
        help="yolo = YOLO11 (быстро), detr = DINOv2+DETR (точнее)",
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)

    parser.add_argument(
        "--model",
        default=None,
        help="Путь к весам YOLO, например yolo11s.pt",
    )

    args = parser.parse_args()

    global YOLO_EPOCHS, YOLO_BATCH, YOLO_MODEL
    global DETR_EPOCHS, DETR_BATCH

    if args.epochs:
        YOLO_EPOCHS = args.epochs
        DETR_EPOCHS = args.epochs

    if args.batch:
        YOLO_BATCH = args.batch
        DETR_BATCH = args.batch

    if args.model:
        YOLO_MODEL = args.model

    if args.mode == "yolo":
        train_yolo()
    else:
        train_detr()


if __name__ == "__main__":
    main()