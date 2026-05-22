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

def create_user(db, first_name, last_name, username, email, password, profile_image=None):
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
    try:
        if pwd_context.verify(plain_password[:72], hashed_password):
            return True
    except Exception:
        pass
    try:
        password_sha = hashlib.sha256(plain_password.encode()).hexdigest()
        if pwd_context.verify(password_sha, hashed_password):
            return True
    except Exception:
        pass
    return False

def authenticate_user(db, username_or_email: str, password: str):
    username_or_email = username_or_email.strip()
    user = db.query(models.User).filter(
        or_(
            models.User.username == username_or_email,
            models.User.email == username_or_email.lower()
        )
    ).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# ══════════════════════════════════════════════
# YÜKLENEN DOSYALAR
# ══════════════════════════════════════════════

def create_uploaded_file(db: Session, filename, saved_path, file_type, file_size, user_id=None):
    record = models.UploadedFile(
        user_id=user_id, filename=filename, saved_path=saved_path,
        file_type=file_type, file_size=file_size,
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

def create_routing_result(db: Session, total_cost, iterations, routes, user_id=None, uploaded_file_id=None):
    record = models.RoutingResult(
        user_id=user_id, uploaded_file_id=uploaded_file_id,
        iterations=iterations, total_cost=total_cost,
        route_count=len(routes),
        routes_json=json.dumps(routes, ensure_ascii=False),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_routing_results(db: Session, user_id: int = None, limit: int = 20):
    q = db.query(models.RoutingResult)
    if user_id:
        q = q.filter(models.RoutingResult.user_id == user_id)
    return q.order_by(models.RoutingResult.created_at.desc()).limit(limit).all()

def get_routing_result(db: Session, result_id: int):
    return db.query(models.RoutingResult).filter(models.RoutingResult.id == result_id).first()


# ══════════════════════════════════════════════
# LSTM SONUÇLARI
# ══════════════════════════════════════════════

def create_lstm_result(db, **kwargs):
    record = models.LSTMResult(**kwargs)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_lstm_results(db: Session, user_id: int = None, limit: int = 20):
    """
    model_data ve scaler_data binary kolonları da çekiliyor,
    böylece geçmiş modellerden indirme çalışır.
    """
    q = db.query(models.LSTMResult)
    if user_id:
        q = q.filter(models.LSTMResult.user_id == user_id)
    return q.order_by(models.LSTMResult.created_at.desc()).limit(limit).all()


# ══════════════════════════════════════════════
# SEGMENTASYON SONUÇLARI
# ══════════════════════════════════════════════

def create_segmentation_result(db, **kwargs):
    record = models.SegmentationResult(**kwargs)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_segmentation_results(db: Session, user_id: int = None, limit: int = 20):
    q = db.query(models.SegmentationResult)
    if user_id:
        q = q.filter(models.SegmentationResult.user_id == user_id)
    return q.order_by(models.SegmentationResult.created_at.desc()).limit(limit).all()
