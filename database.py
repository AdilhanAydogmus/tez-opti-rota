# database.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./tez_app.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("postgresql"):
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"connect_timeout": 10}
    )
else:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from models import Base
    Base.metadata.create_all(bind=engine)
    _run_migrations()  # init_db içinde çağır, modül seviyesinde değil


def _run_migrations():
    """PostgreSQL'e özgü kolon eklemeleri — sadece PostgreSQL'de çalıştır."""
    if not DATABASE_URL.startswith("postgresql"):
        return  # SQLite'ta atla, create_all zaten halleder

    sqls = [
        "ALTER TABLE lstm_results ADD COLUMN IF NOT EXISTS loss_plot_b64 TEXT",
        "ALTER TABLE lstm_results ADD COLUMN IF NOT EXISTS prediction_plot_b64 TEXT",
        "ALTER TABLE lstm_results ADD COLUMN IF NOT EXISTS model_data BYTEA",
        "ALTER TABLE lstm_results ADD COLUMN IF NOT EXISTS scaler_data BYTEA",
        "ALTER TABLE lstm_results ADD COLUMN IF NOT EXISTS loss_plot_path VARCHAR(500)",
        "ALTER TABLE lstm_results ADD COLUMN IF NOT EXISTS prediction_plot_path VARCHAR(500)",
        "ALTER TABLE lstm_results ADD COLUMN IF NOT EXISTS model_path VARCHAR(500)",
        "ALTER TABLE lstm_results ADD COLUMN IF NOT EXISTS scaler_path VARCHAR(500)",
    ]
    try:
        with engine.connect() as conn:
            for sql in sqls:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                except Exception as e:
                    print(f"Migration (görmezden gelindi): {e}")
    except Exception as e:
        print(f"Migration bağlantı hatası (görmezden gelindi): {e}")
