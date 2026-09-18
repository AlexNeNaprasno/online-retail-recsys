"""
Очистка Online Retail dataset (Kaggle: carrie1/ecommerce-data).

Логика:
1. Убираем отмены (InvoiceNo начинается с 'C').
2. Убираем возвраты и корректировки (Quantity <= 0).
3. Убираем ошибки цены (UnitPrice <= 0).
4. Убираем гостевые покупки (CustomerID = NaN).
5. Убираем служебные StockCode (POST, DOT, M, BANK CHARGES, ...).
6. Парсим InvoiceDate в datetime.
7. CustomerID -> int.
8. Заполняем пропуски Description по StockCode.
9. Удаляем полные дубликаты.
10. Добавляем TotalPrice = Quantity * UnitPrice.
11. Сортируем по (CustomerID, InvoiceDate) — важно для последовательностей!

Результат: data/processed/retail_clean.parquet
"""
from pathlib import Path

import pandas as pd

# ------- Пути -------
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "data.csv"
OUT_PATH = ROOT / "data" / "processed" / "retail_clean.parquet"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Служебные StockCode — это не товары, а услуги/сборы.
# Их нужно убрать, иначе они попадут в словарь товаров как «псевдо-товары».
SERVICE_CODES = {
    "POST", "DOT", "M", "BANK CHARGES", "AMAZONFEE",
    "C2", "D", "S", "TEST", "ADJUST", "B", "CRUK",
    "PADS", "SP1002", "BANK CHARGES",
}


def load_raw() -> pd.DataFrame:
    """Загружаем сырой CSV."""
    print(f"Загружаю сырые данные: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, encoding="ISO-8859-1")
    print(f"  Строк: {len(df):,}")
    print(f"  Колонки: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн очистки. Возвращает чистый DataFrame."""
    print("\nНачинаю очистку...")
    n_start = len(df)

    # ---- Шаг 1. Убираем отмены (InvoiceNo начинается с 'C') ----
    n_before = len(df)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    print(f"  1. Убрал отмены:               {n_before - len(df):>7,}  строк -> осталось {len(df):,}")

    # ---- Шаг 2. Quantity > 0 ----
    n_before = len(df)
    df = df[df["Quantity"] > 0]
    print(f"  2. Убрал Quantity <= 0:        {n_before - len(df):>7,}  строк -> осталось {len(df):,}")

    # ---- Шаг 3. UnitPrice > 0 ----
    n_before = len(df)
    df = df[df["UnitPrice"] > 0]
    print(f"  3. Убрал UnitPrice <= 0:       {n_before - len(df):>7,}  строк -> осталось {len(df):,}")

    # ---- Шаг 4. CustomerID не NaN ----
    n_before = len(df)
    df = df[df["CustomerID"].notna()]
    print(f"  4. Убрал NaN CustomerID:       {n_before - len(df):>7,}  строк -> осталось {len(df):,}")

    # ---- Шаг 5. Убираем служебные StockCode ----
    n_before = len(df)
    df = df[~df["StockCode"].astype(str).str.upper().isin(SERVICE_CODES)]
    print(f"  5. Убрал служебные StockCode:  {n_before - len(df):>7,}  строк -> осталось {len(df):,}")

    # ---- Шаг 6. Парсим дату ----
    print("  6. Парсю InvoiceDate...")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=False)

    # ---- Шаг 7. CustomerID -> int ----
    df["CustomerID"] = df["CustomerID"].astype(int)

    # ---- Шаг 8. Заполняем пропуски Description по StockCode ----
    # Идея: у каждого StockCode самое частое описание.
    # Если у товара пропущено описание — подставим моду по этому StockCode.
    print("  8. Заполняю пропуски Description по StockCode...")
    desc_map = (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    df["Description"] = df["Description"].fillna(df["StockCode"].map(desc_map))
    # Если всё ещё NaN (значит, у этого StockCode вообще нет описаний) — просто убираем
    df = df.dropna(subset=["Description"])

    # ---- Шаг 9. Удаляем полные дубликаты ----
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  9. Убрал дубликаты:            {n_before - len(df):>7,}  строк -> осталось {len(df):,}")

    # ---- Шаг 10. TotalPrice ----
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # ---- Шаг 11. Сортировка ----
    # КРИТИЧНО для рекомендаций: последовательности должны быть
    # упорядочены по времени для каждого клиента.
    df = df.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)

    print(f"\nОчистка завершена: {n_start:,} -> {len(df):,} строк")
    print(f"  Уникальных клиентов: {df['CustomerID'].nunique():,}")
    print(f"  Уникальных товаров:  {df['StockCode'].nunique():,}")
    print(f"  Диапазон дат:        {df['InvoiceDate'].min().date()} .. {df['InvoiceDate'].max().date()}")
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    """Сохраняем в Parquet."""
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"\nСохранил в {path} ({size_mb:.1f} МБ)")


def main():
    df = load_raw()
    df = clean(df)
    save(df, OUT_PATH)


if __name__ == "__main__":
    main()