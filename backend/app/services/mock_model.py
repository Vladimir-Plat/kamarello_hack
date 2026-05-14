import random
import time
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    'filename','product_name','price_default','price_card','price_discount','barcode','discount_amount','id_sku',
    'print_datetime','code','additional_info','color','special_symbols','frame_timestamp','x_min','y_min','x_max','y_max',
    'qr_code_barcode','price1_qr','price2_qr','price3_qr','price4_qr','wholesale_level_1_count','wholesale_level_1_price',
    'wholesale_level_2_count','wholesale_level_2_price','action_price_qr','action_code_qr'
]

DEMO_PRODUCTS = [
    ('Молоко ЛЕНТА пастеризованное 2.5% 930 мл', '4607004890012', 89.99, 74.99, 'blue'),
    ('Йогурт питьевой клубника 270 г', '4607084123120', 69.99, 59.99, 'red'),
    ('Сыр полутвердый Российский 45% 200 г', '4607012345678', 249.99, 199.99, 'yellow'),
    ('Мед натуральный липовый 500 г', '4603552017456', 415.79, 316.99, 'red'),
    ('Напиток безалкогольный SANTO STEFANO 0.25L', '4670025474665', 252.63, 129.99, 'red'),
]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'wholesale_level_1_coun': 'wholesale_level_1_count'})
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    return df[REQUIRED_COLUMNS]

def _find_sample_for_video(video_name: str, sample_dir: Path) -> Path | None:
    stem = Path(video_name).stem
    candidates = list(sample_dir.rglob(f'{stem}.csv'))
    if candidates:
        return candidates[0]
    all_csv = [p for p in sample_dir.rglob('*.csv') if p.name != 'sample.csv']
    return random.choice(all_csv) if all_csv else None

def analyze_video(video_path: Path, sample_dir: Path, progress_callback=None) -> pd.DataFrame:
    # Stub for future CV/OCR pipeline. Replace this function with a real neural network adapter.
    for p in (10, 25, 45, 70, 90):
        time.sleep(0.25)
        if progress_callback:
            progress_callback(p)

    sample = _find_sample_for_video(video_path.name, sample_dir)
    if sample:
        df = pd.read_csv(sample)
        df = _normalize_columns(df)
        df['filename'] = video_path.name
        return df

    rows = []
    for i in range(random.randint(8, 16)):
        name, barcode, default, card, color = random.choice(DEMO_PRODUCTS)
        x = random.randint(120, 3200)
        y = random.randint(400, 1900)
        rows.append({
            'filename': video_path.name,
            'product_name': name,
            'price_default': default,
            'price_card': card,
            'price_discount': 'нет',
            'barcode': barcode,
            'discount_amount': f'-{round((1-card/default)*100)}%',
            'id_sku': str(random.randint(270000000000, 370999999999)),
            'print_datetime': 'нет',
            'code': 'нет',
            'additional_info': 'нет',
            'color': color,
            'special_symbols': 'нет',
            'frame_timestamp': random.randint(0, 120000),
            'x_min': x, 'y_min': y, 'x_max': x + random.randint(120, 260), 'y_max': y + random.randint(90, 220),
            'qr_code_barcode': barcode,
            'price1_qr': default,
            'price2_qr': 'нет',
            'price3_qr': 'нет',
            'price4_qr': card,
            'wholesale_level_1_count': 'нет',
            'wholesale_level_1_price': 'нет',
            'wholesale_level_2_count': 'нет',
            'wholesale_level_2_price': 'нет',
            'action_price_qr': 'нет',
            'action_code_qr': 'нет',
        })
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
