import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)


def get_engine():
    """Возвращает SQLAlchemy engine для retail_db."""
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")

    missing = [k for k, v in {
        "DB_USER": user, "DB_PASSWORD": password,
        "DB_HOST": host, "DB_PORT": port, "DB_NAME": db,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Не загружены переменные: {missing}")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)