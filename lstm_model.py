import os
import io
import base64
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from veri_on_isleme import veri_on_islem


def _fig_to_b64(fig) -> str:
    """Matplotlib figure'ı base64 PNG string'e çevirir."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def lstm_model_egit(data, window_size=30, epochs=50, test_ratio=0.10,
                    output_dir="outputs", plot_dir="static/plots"):
    """
    Modeli eğitir. Grafikler ve model binary olarak döner (dosyaya yazmaz).
    output_dir ve plot_dir parametreleri artık kullanılmaz, geriye dönük uyumluluk için tutuldu.
    """

    X_train, X_test, y_train, y_test, scaler, df_clean = veri_on_islem(
        data, window_size=window_size, test_ratio=test_ratio
    )

    split_val = int(len(X_test) * 0.5)
    if split_val == 0 or len(X_test) - split_val == 0:
        raise ValueError("Validation/Test ayırımı için yeterli test verisi yok.")

    X_val, y_val = X_test[:split_val], y_test[:split_val]
    X_test, y_test = X_test[split_val:], y_test[split_val:]

    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test, verbose=0)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler.inverse_transform(y_pred)

    y_test_real = np.nan_to_num(y_test_real, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred_real = np.nan_to_num(y_pred_real, nan=0.0, posinf=0.0, neginf=0.0)
    y_test_real = np.maximum(0, y_test_real).flatten()
    y_pred_real = np.maximum(0, y_pred_real).flatten()

    mae  = float(mean_absolute_error(y_test_real, y_pred_real))
    rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
    mask = y_test_real != 0
    mape = float(np.mean(np.abs((y_test_real[mask] - y_pred_real[mask]) / y_test_real[mask])) * 100)

    # ── Loss grafiği → base64 ──
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history.history["loss"], label="Eğitim Loss")
    ax1.plot(history.history["val_loss"], label="Doğrulama Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Huber Loss")
    ax1.set_title("LSTM Eğitim ve Doğrulama Loss Grafiği")
    ax1.legend(); ax1.grid(True)
    fig1.tight_layout()
    loss_plot_b64 = _fig_to_b64(fig1)

    # ── Tahmin grafiği → base64 ──
    n_plot = min(200, len(y_test_real))
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(y_test_real[:n_plot], label="Gerçek Talep")
    ax2.plot(y_pred_real[:n_plot], label="Tahmin Edilen Talep")
    ax2.set_xlabel("Gözlem"); ax2.set_ylabel("Talep")
    ax2.set_title("Gerçek ve Tahmin Edilen Talep Karşılaştırması")
    ax2.legend(); ax2.grid(True)
    fig2.tight_layout()
    prediction_plot_b64 = _fig_to_b64(fig2)

    # ── Model → binary ──
    import tempfile
    _tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    _tmp.close()
    model.save(_tmp.name)
    with open(_tmp.name, "rb") as f:
        model_data = f.read()
    os.remove(_tmp.name)

    # ── Scaler → binary ──
    scaler_buf = io.BytesIO()
    joblib.dump(scaler, scaler_buf)
    scaler_data = scaler_buf.getvalue()

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "epoch_count": int(len(history.history["loss"])),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "loss_plot_b64": loss_plot_b64,
        "prediction_plot_b64": prediction_plot_b64,
        "model_data": model_data,
        "scaler_data": scaler_data,
        # Geriye dönük uyumluluk
        "loss_plot": "db",
        "prediction_plot": "db",
        "model_path": None,
        "scaler_path": None,
    }
