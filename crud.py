
from sqlalchemy import or_
"""
crud.py — Veritabanı okuma/yazma işlemleri.
"""
import hashlib
import json
from sqlalchemy.orm import Session
from passlib.context import CryptContext

import models

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ══════════════════════════════════════════════
# KULLANICI
# ══════════════════════════════════════════════

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def create_user(
    db,
    first_name,
    last_name,
    username,
    email,
    password,
    profile_image=None
):
    # Yeni kullanıcılar için direkt bcrypt hash
    hashed_password = pwd_context.hash(password[:72])

    user = models.User(
        first_name=first_name.strip(),
        last_name=last_name.strip(),
        username=username.strip(),
        email=email.strip().lower(),
        hashed_password=hashed_password,
        profile_image=profile_image
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Hem yeni bcrypt şifrelerini hem de eski SHA256+bcrypt
    yapısını destekler.
    """

    try:
        # Yeni sistem
        if pwd_context.verify(plain_password[:72], hashed_password):
            return True
    except Exception:
        pass

    try:
        # Eski sistem desteği
        password_sha = hashlib.sha256(
            plain_password.encode()
        ).hexdigest()

        if pwd_context.verify(password_sha, hashed_password):
            return True
    except Exception:
        pass

    return False


def authenticate_user(
    db,
    username_or_email: str,
    password: str
):
    username_or_email = username_or_email.strip()

    user = db.query(models.User).filter(
        or_(
            models.User.username == username_or_email,
            models.User.email == username_or_email.lower()
        )
    ).first()

    if not user:
        return None

    if not verify_password(
        password,
        user.hashed_password
    ):
        return None

    return user


# ══════════════════════════════════════════════
# YÜKLENEN DOSYALAR
# ══════════════════════════════════════════════

def create_uploaded_file(
    db: Session,
    filename: str,
    saved_path: str,
    file_type: str,
    file_size: int,
    user_id: int = None,
):
    record = models.UploadedFile(
        user_id=user_id,
        filename=filename,
        saved_path=saved_path,
        file_type=file_type,
        file_size=file_size,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_uploaded_files(db: Session, user_id: int = None, limit: int = 50):
    q = db.query(models.UploadedFile)
    if user_id:
        q = q.filter(models.UploadedFile.user_id == user_id)
    return q.order_by(models.UploadedFile.uploaded_at.desc()).limit(limit).all()


# ══════════════════════════════════════════════
# ROTALAMA SONUÇLARI
# ══════════════════════════════════════════════

def create_routing_result(
    db: Session,
    total_cost: float,
    iterations: int,
    routes: list,
    user_id: int = None,
    uploaded_file_id: int = None,
):
    record = models.RoutingResult(
        user_id=user_id,
        uploaded_file_id=uploaded_file_id,
        iterations=iterations,
        total_cost=total_cost,
        route_count=len(routes),
        routes_json=json.dumps(routes, ensure_ascii=False),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_routing_results(db: Session, user_id: int = None, limit: int = 20):
    q = db.query(
        models.RoutingResult.id,
        models.RoutingResult.user_id,
        models.RoutingResult.uploaded_file_id,
        models.RoutingResult.iterations,
        models.RoutingResult.total_cost,
        models.RoutingResult.route_count,
        models.RoutingResult.routes_json,
        models.RoutingResult.created_at,
    )
    if user_id:
        q = q.filter(models.RoutingResult.user_id == user_id)
    rows = q.order_by(models.RoutingResult.created_at.desc()).limit(limit).all()

    class _Row:
        pass

    results = []
    for r in rows:
        obj = _Row()
        obj.id = r.id
        obj.user_id = r.user_id
        obj.uploaded_file_id = r.uploaded_file_id
        obj.iterations = r.iterations
        obj.total_cost = r.total_cost
        obj.route_count = r.route_count
        obj.routes_json = r.routes_json
        obj.created_at = r.created_at
        results.append(obj)
    return results

def get_routing_result(db: Session, result_id: int):
    return db.query(models.RoutingResult).filter(
        models.RoutingResult.id == result_id
    ).first()


def create_lstm_result(db, **kwargs):
    record = models.LSTMResult(**kwargs)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_lstm_results(db: Session, user_id: int = None, limit: int = 20):
    # Sadece kesinlikle var olan kolonları seç (eski DB şemasıyla uyumluluk için)
    q = db.query(
        models.LSTMResult.id,
        models.LSTMResult.user_id,
        models.LSTMResult.uploaded_file_id,
        models.LSTMResult.epochs,
        models.LSTMResult.window_size,
        models.LSTMResult.mae,
        models.LSTMResult.rmse,
        models.LSTMResult.mape,
        models.LSTMResult.final_loss,
        models.LSTMResult.final_val_loss,
        models.LSTMResult.epoch_count,
        models.LSTMResult.created_at,
    )

    if user_id:
        q = q.filter(models.LSTMResult.user_id == user_id)

    rows = q.order_by(
        models.LSTMResult.created_at.desc()
    ).limit(limit).all()

    # Row'ları dict benzeri nesnelere çevir (endpoint uyumluluğu için)
    class _Row:
        pass

    results = []
    for r in rows:
        obj = _Row()
        obj.id = r.id
        obj.user_id = r.user_id
        obj.uploaded_file_id = r.uploaded_file_id
        obj.epochs = r.epochs
        obj.window_size = r.window_size
        obj.mae = r.mae
        obj.rmse = r.rmse
        obj.mape = r.mape
        obj.final_loss = r.final_loss
        obj.final_val_loss = r.final_val_loss
        obj.epoch_count = r.epoch_count
        obj.created_at = r.created_at
        obj.model_path = None
        obj.scaler_path = None
        obj.model_data = None
        obj.scaler_data = None
        results.append(obj)

    return results


def create_segmentation_result(db, **kwargs):
    record = models.SegmentationResult(**kwargs)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_segmentation_results(db: Session, user_id: int = None, limit: int = 20):
    q = db.query(
        models.SegmentationResult.id,
        models.SegmentationResult.user_id,
        models.SegmentationResult.uploaded_file_id,
        models.SegmentationResult.n_clusters,
        models.SegmentationResult.best_k,
        models.SegmentationResult.silhouette_score,
        models.SegmentationResult.window_size,
        models.SegmentationResult.cluster_summary_json,
        models.SegmentationResult.created_at,
    )

    if user_id:
        q = q.filter(models.SegmentationResult.user_id == user_id)

    rows = q.order_by(
        models.SegmentationResult.created_at.desc()
    ).limit(limit).all()

    class _Row:
        pass

    results = []
    for r in rows:
        obj = _Row()
        obj.id = r.id
        obj.user_id = r.user_id
        obj.uploaded_file_id = r.uploaded_file_id
        obj.n_clusters = r.n_clusters
        obj.best_k = r.best_k
        obj.silhouette_score = r.silhouette_score
        obj.window_size = r.window_size
        obj.cluster_summary_json = r.cluster_summary_json
        obj.created_at = r.created_at
        obj.excel_data = None  # binary kolon — ayrı sorguyla çekiliyor
        results.append(obj)
    return results
