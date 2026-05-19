# 🏷️ Lenta Price Tag Detector

Три файла. Два режима: **YOLO11** (быстро) и **DINOv2+DETR** (точнее).

---

## Структура проекта

```
lenta_best/
├── setup.bat              ← установка (запустить один раз)
├── requirements.txt
├── prepare.py             ← Шаг 1: CSV → YOLO + COCO датасет
├── train.py               ← Шаг 2: обучение YOLO11 или DETR
├── infer.py               ← Шаг 3: inference → crops + CSV
└── input_videos/
    ├── video_001.mp4      ← ваше видео
    └── video_001.csv      ← разметка (то же имя!)
```

**Создаются автоматически:**
```
annotations/               ← промежуточный JSON (для отладки)
debug_vis/                 ← (если --debug) кадры с нарисованными bbox
dataset/
  images/train, val/       ← кадры
  labels/train, val/       ← YOLO labels
  annotations/             ← COCO JSON для DETR
  dataset.yaml             ← конфиг для YOLO
runs/train/
  yolo11/weights/best.pt   ← лучшие веса YOLO11
  detr/best.pt             ← лучшие веса DETR
output/
  crops/                   ← вырезанные ценники
  results.csv              ← итоговая таблица
```

---

## Установка

1. Python 3.10 или 3.11 с [python.org](https://python.org) (галочка "Add to PATH")
2. Запустить `setup.bat`
3. После установки всегда активировать окружение:
   ```
   .venv\Scripts\activate
   ```

---

## Формат входного CSV

Файл CSV должен называться **так же, как видео**:

```
input_videos/video_001.mp4
input_videos/video_001.csv
```

Содержимое CSV:
```csv
frame_timestamp,x_min,y_min,x_max,y_max
1000,142,88,352,183
2500,45,210,280,305
```

| Поле | Описание |
|------|----------|
| `frame_timestamp` | Время кадра в **миллисекундах** от начала видео |
| `x_min, y_min` | Левый верхний угол bbox (пиксели) |
| `x_max, y_max` | Правый нижний угол bbox (пиксели) |

> Поддерживается запятая как разделитель дробной части: `142,5` = `142.5`

---

## Запуск: пошаговая инструкция

### Шаг 1 — Подготовка датасета

```bash
python prepare.py

# Опции:
python prepare.py --debug     # сохранить debug_vis/ с нарисованными bbox
python prepare.py --no-aug    # без аугментации (для быстрой проверки)
```

**Что делает:**
- Читает все CSV из `input_videos/`
- Конвертирует timestamp → извлекает кадры из видео
- Сохраняет промежуточный JSON в `annotations/` (для отладки)
- Применяет 6 видов аугментации × 3 копии каждого кадра
- Экспортирует в YOLO format и COCO JSON (для DETR)

**Проверка результата:**
```bash
python prepare.py --debug
# Смотрите debug_vis/ — там кадры с нарисованными bbox
# Если bbox не там — проверьте annotations/video.json
```

---

### Шаг 2 — Обучение

#### Режим A: YOLO11 (рекомендуется для начала)

```bash
python train.py

# С параметрами:
python train.py --epochs 50 --batch 8
python train.py --model yolo11s.pt    # s/m/l/x для лучшего качества
```

- Скачивает `yolo11n.pt` (~6 МБ) автоматически
- Обучает 100 эпох с early stopping
- RTX 4060: ~10-20 мин

Результат: `runs/train/yolo11/weights/best.pt`

#### Режим B: DINOv2 + DETR (для лучшей точности)

```bash
python train.py --mode detr

# С параметрами:
python train.py --mode detr --epochs 30 --batch 2
```

- Скачивает DETR + DINOv2 веса (~500 МБ) при первом запуске
- RTX 4060: ~30-60 мин

Результат: `runs/train/detr/best.pt`

---

### Шаг 3 — Inference

```bash
# Все видео из input_videos/ (batch)
python infer.py
python infer.py --mode detr      # для DETR режима

# Одно видео
python infer.py --source input_videos/video_001.mp4

# С визуализацией на экране (для отладки)
python infer.py --source input_videos/video_001.mp4 --preview

# Webcam
python infer.py --source 0 --preview

# Изменить порог уверенности
python infer.py --conf 0.15      # ниже = больше ценников, меньше пропусков
python infer.py --conf 0.45      # выше = меньше ложных срабатываний

# Посчитать метрики на val датасете (только YOLO)
python infer.py --eval
```

---

## Метрики качества

```
python infer.py --eval
```

Выводит:
```
  mAP@0.5:      0.8742
  mAP@0.5:0.95: 0.6231
  Precision:    0.9012
  Recall:       0.8543
```

**Recall важнее Precision** для данной задачи:
пропущенный ценник = потерянные данные о цене.
Рекомендуемое значение Recall: ≥ 0.85.

---

## Настройки в начале файлов

### train.py
```python
YOLO_MODEL    = "yolo11n.pt"  # n/s/m/l/x — размер модели
YOLO_EPOCHS   = 100
YOLO_BATCH    = 16            # уменьшить до 8 при OOM
DETR_EPOCHS   = 50
DETR_BATCH    = 4             # уменьшить до 2 при OOM
```

### infer.py
```python
CONF_THRESHOLD = 0.25   # порог уверенности
MAX_MISSED     = 30     # кадров без детекции до завершения трека
```

### prepare.py
```python
AUG_FACTOR = 3          # копий аугментации на каждый кадр
VAL_SPLIT  = 0.2        # 20% на валидацию
MIN_BOX_PX = 15         # минимальный bbox в пикселях
```

---

## Как работает выбор лучшего кадра

Для каждого ценника трекер следит за ним всё время пока он в кадре.
Лучший кадр выбирается по формуле:

```
quality = 0.40 × резкость (дисперсия Лапласиана, норм.)
        + 0.30 × confidence модели
        + 0.20 × площадь bbox / площадь кадра
        + 0.10 × соотношение сторон (ценники обычно 2:1 – 4:1)
```

Даже полностью размытый ценник сохраняется — берётся лучший доступный вариант.

---

## Формат выходного CSV

| Поле | Описание |
|------|----------|
| `filename` | Имя видеофайла |
| `track_id` | ID трекера |
| `unique_id` | Глобальный UUID ценника |
| `frame_timestamp_ms` | Время лучшего кадра (мс) |
| `x_min/y_min/x_max/y_max` | Координаты bbox |
| `confidence` | Уверенность модели 0..1 |
| `sharpness` | Дисперсия Лапласиана (резкость) |
| `quality_score` | Итоговый скор 0..1 |
| `crop_file` | Имя файла изображения ценника |

---

## Устранение проблем

| Проблема | Решение |
|----------|---------|
| `CUDA out of memory` | `YOLO_BATCH = 8` или `DETR_BATCH = 2` |
| CSV столбцы не найдены | Проверить заголовок: `frame_timestamp,x_min,y_min,x_max,y_max` |
| Кадры не извлекаются | Смотреть `annotations/video.json` — правильные ли timestamp |
| Bbox не там | `python prepare.py --debug` → смотреть `debug_vis/` |
| Пропускает ценники | `--conf 0.15` или увеличить `AUG_FACTOR = 5` |
| Много ложных | `--conf 0.45` |
| Видео не открывается | Установить ffmpeg: `winget install ffmpeg` |
| Модель не скачивается | Проверить интернет; файлы кешируются в `~/.cache/` |

---

## Быстрый старт (TL;DR)

```bash
# 1. Установка
setup.bat
.venv\Scripts\activate

# 2. Положить видео + CSV в input_videos\

# 3. Три шага
python prepare.py --debug    # проверьте debug_vis/
python train.py              # YOLO11
python infer.py --eval       # метрики
python infer.py --preview    # визуальная проверка
```
