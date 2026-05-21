from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    Text, ForeignKey, Boolean, LargeBinary
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
from datetime import datetime, timezone, timedelta

# Türkiye saati (UTC+3)
def turkey_now():
    return datetime.now(timezone(timedelta(hours=3)))


# ─────────────────────────────────────────────
# KULLANICI
# ─────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    profile_image = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=turkey_now)

    uploaded_files = relationship("UploadedFile", back_populates="user")
    routing_results = relationship("RoutingResult", back_populates="user")
    lstm_results = relationship("LSTMResult", back_populates="user")
    segmentation_results = relationship("SegmentationResult", back_populates="user")

# ─────────────────────────────────────────────
# YÜKLENEN DOSYALAR
# ─────────────────────────────────────────────

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    filename    = Column(String(255), nullable=False)
    saved_path  = Column(String(500), nullable=False)
    file_type   = Column(String(50))
    file_size   = Column(Integer)
    uploaded_at = Column(DateTime(timezone=True), default=turkey_now)

    user             = relationship("User", back_populates="uploaded_files")
    routing_results  = relationship("RoutingResult",      back_populates="uploaded_file")
    lstm_results     = relationship("LSTMResult",         back_populates="uploaded_file")
    seg_results      = relationship("SegmentationResult", back_populates="uploaded_file")


# ─────────────────────────────────────────────
# ROTALAMA SONUÇLARI
# ─────────────────────────────────────────────

class RoutingResult(Base):
    __tablename__ = "routing_results"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=True)
    uploaded_file_id = Column(Integer, ForeignKey("uploaded_files.id"), nullable=True)
    iterations       = Column(Integer)
    total_cost       = Column(Float)
    route_count      = Column(Integer)
    routes_json      = Column(Text)
    created_at       = Column(DateTime(timezone=True), default=turkey_now)

    user          = relationship("User",         back_populates="routing_results")
    uploaded_file = relationship("UploadedFile", back_populates="routing_results")


# ─────────────────────────────────────────────
# LSTM MODEL SONUÇLARI
# ─────────────────────────────────────────────

class LSTMResult(Base):
    __tablename__ = "lstm_results"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=True)
    uploaded_file_id = Column(Integer, ForeignKey("uploaded_files.id"), nullable=True)
    epochs           = Column(Integer)
    window_size      = Column(Integer)
    mae              = Column(Float)
    rmse             = Column(Float)
    mape             = Column(Float)
    final_loss       = Column(Float)
    final_val_loss   = Column(Float)
    epoch_count      = Column(Integer)

    # Grafikler base64 olarak DB'de
    loss_plot_b64        = Column(Text, nullable=True)
    prediction_plot_b64  = Column(Text, nullable=True)

    # Model ve scaler binary olarak DB'de
    model_data           = Column(LargeBinary, nullable=True)
    scaler_data          = Column(LargeBinary, nullable=True)

    # Geriye dönük uyumluluk için path alanları (boş kalacak)
    loss_plot_path        = Column(String(500), nullable=True)
    prediction_plot_path  = Column(String(500), nullable=True)
    model_path            = Column(String(500), nullable=True)
    scaler_path           = Column(String(500), nullable=True)

    created_at       = Column(DateTime(timezone=True), default=turkey_now)

    user          = relationship("User",         back_populates="lstm_results")
    uploaded_file = relationship("UploadedFile", back_populates="lstm_results")


# ─────────────────────────────────────────────
# SEGMENTASYON SONUÇLARI
# ─────────────────────────────────────────────

class SegmentationResult(Base):
    __tablename__ = "segmentation_results"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=True)
    uploaded_file_id = Column(Integer, ForeignKey("uploaded_files.id"), nullable=True)
    n_clusters        = Column(Integer)
    best_k            = Column(Integer)
    silhouette_score  = Column(Float)
    window_size       = Column(Integer)

    # Grafikler base64 olarak DB'de
    cluster_plot_b64     = Column(Text, nullable=True)
    silhouette_plot_b64  = Column(Text, nullable=True)

    # Excel binary olarak DB'de
    excel_data           = Column(LargeBinary, nullable=True)

    # Geriye dönük uyumluluk
    cluster_plot_path     = Column(String(500), nullable=True)
    silhouette_plot_path  = Column(String(500), nullable=True)
    excel_output_path     = Column(String(500), nullable=True)
    cluster_summary_json  = Column(Text)

    created_at       = Column(DateTime(timezone=True), default=turkey_now)

    user          = relationship("User",         back_populates="segmentation_results")
    uploaded_file = relationship("UploadedFile", back_populates="seg_results")
