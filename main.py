from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, Form, BackgroundTasks
import threading
import json
from typing import Annotated, Any, Optional
import os
import uuid
import shutil
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel as PydanticBase

import crud
import models
import database
from database import get_db, init_db
from auth import get_current_user, get_optional_user, create_access_token

from sezgisel import VRPData, ALNSSetPartitioning
from kumeleme import kumeleme_pipeline
from lstm_model import lstm_model_egit
from fastapi.responses import (
    HTMLResponse,
    FileResponse,
    JSONResponse
)

# ── LSTM BACKGROUND TASK STATE ───────────────
_lstm_tasks: dict = {}  # task_id -> durum dict

app = FastAPI(title="ARAÇ ROTALAMA, LSTM VE MÜŞTERİ SEGMENTASYONU")

@app.on_event("startup")
async def startup():
    init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("outputs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)
os.makedirs("static/maps", exist_ok=True)


class RotalamaSonuc(BaseModel):
    dosya_adi: str
    iterasyon: int
    rota: Any
    maliyet: float


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/lstm/tez-sonuc")
async def lstm_tez_sonuc():
    return {
        "mae": 0.0821,
        "rmse": 0.1143,
        "mape": 8.42,
        "epoch_count": 3,
        "final_loss": 0.0067,
        "final_val_loss": 0.0089,
        "loss_plot_url": "/static/plots/lstm_loss.png",
        "prediction_plot_url": "/static/plots/lstm_prediction.png",
    }


@app.get("/rotalama", response_class=HTMLResponse)
async def rotalama_page(request: Request, current_user: models.User = Depends(get_optional_user)):
    return templates.TemplateResponse(
        request=request,
        name="rotalama.html",
        context={"request": request, "current_user": current_user}
    )


@app.get("/hakkimizda", response_class=HTMLResponse, name="hakkimizda")
async def hakkimizda_sayfasi(request: Request, current_user: models.User = Depends(get_optional_user)):
    return templates.TemplateResponse(
        request=request,
        name="hakkimizda.html",
        context={"request": request, "current_user": current_user}
    )


@app.get("/lstm-egitimi", response_class=HTMLResponse, name="lstm_egitimi")
async def lstm_egitimi_sayfasi(request: Request, current_user: models.User = Depends(get_optional_user)):
    return templates.TemplateResponse(
        request=request,
        name="lstm_egitimi.html",
        context={"request": request, "current_user": current_user}
    )


@app.get("/musteri-segmentasyonu", response_class=HTMLResponse, name="musteri_segmentasyonu")
async def musteri_segmentasyonu_sayfasi(request: Request, current_user: models.User = Depends(get_optional_user)):
    return templates.TemplateResponse(
        request=request,
        name="musteri_segmentasyonu.html",
        context={"request": request, "current_user": current_user}
    )


# =========================================================
# TEZ VERİLERİ İLE HAZIR ROTALAMA SONUCU
# =========================================================

@app.api_route("/rotalama/tez", methods=["GET", "POST"])
async def rotalama_tez(current_user: models.User = Depends(get_optional_user)):
    try:
        import pandas as pd
        import ast

        excel_path = "static/rotalama_sonuclari/matheuristic_sonuc.xlsx"
        if not os.path.exists(excel_path):
            return JSONResponse(status_code=404, content={"error": "Dosya bulunamadı"})

        df = pd.read_excel(excel_path)
        routes = []
        total_cost = 0

        for _, row in df.iterrows():
            route_str = str(row["route"])
            route_list = [x.strip() for x in route_str.split("->")]

            koordinatlar = ast.literal_eval(str(row["coordinates"]))
            coordinates = []
            for idx, coord in enumerate(koordinatlar):
                node_id = route_list[idx] if idx < len(route_list) else "0"
                coordinates.append({
                    "id": str(node_id),
                    "lat": float(coord[0]),
                    "lng": float(coord[1])
                })

            route_cost = float(str(row["total_cost"]).replace(",", "."))
            distance_cost = float(str(row["distance_cost"]).replace(",", "."))
            late_cost = float(str(row["late_cost"]).replace(",", "."))
            total_cost += route_cost

            routes.append({
                "service": str(row["service"]),
                "vehicle": str(row["vehicle"]),
                "route": route_list,
                "coordinates": coordinates,
                "distance_cost": distance_cost,
                "late_cost": late_cost,
                "total_cost": route_cost
            })

        return {"total_cost": total_cost, "routes": routes, "download_excel": "/download-rotalama"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tez rotalama sonucu okunamadı: {str(e)}")


# =========================================================
# KENDİ VERİLERİM İLE ROTALAMA
# =========================================================

@app.post("/rotalama/", response_model=RotalamaSonuc, summary="Araç Rotalarını Optimize Et")
async def rotalama_motoru(
    data: Annotated[UploadFile, File()],
    iterations: int = 500,
    current_user: models.User = Depends(get_current_user)
):
    if not data.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Lütfen sadece .xlsx veya .xls dosyası yükleyin.")

    try:
        file_id = str(uuid.uuid4())
        veri_path = os.path.join("uploads", f"{file_id}_{data.filename}")
        with open(veri_path, "wb") as buffer:
            shutil.copyfileobj(data.file, buffer)

        segment_path = "static/veriler/musteri_kumeleme_sonuclari.xlsx"
        data_obj = VRPData(filepath=veri_path, segment_filepath=segment_path)
        solver = ALNSSetPartitioning(data=data_obj, seed=42)
        sonuc = solver.solve(iterations=iterations)

        return {
            "dosya_adi": data.filename,
            "iterasyon": iterations,
            "rota": sonuc["routes"],
            "maliyet": float(sonuc["total_cost"]),
            "download_excel": "/download/matheuristic_sonuc.xlsx"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rotalama sırasında hata oluştu: {str(e)}")


@app.post("/rotalama/kendi-verilerim")
async def rotalama_kendi_verilerim(
    veri_dosyasi: Annotated[UploadFile, File()],
    kume_dosyasi: Optional[UploadFile] = File(default=None),
    iterations: int = Form(default=500),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if not veri_dosyasi.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Rotalama veri dosyası Excel olmalıdır.")

    try:
        file_id = str(uuid.uuid4())
        veri_path = os.path.join("uploads", f"{file_id}_{veri_dosyasi.filename}")
        with open(veri_path, "wb") as buffer:
            shutil.copyfileobj(veri_dosyasi.file, buffer)

        try:
            db_file = crud.create_uploaded_file(
                db=db, filename=veri_dosyasi.filename, saved_path=veri_path,
                file_type="rotalama", file_size=os.path.getsize(veri_path), user_id=current_user.id
            )
        except Exception as db_error:
            print("DB DOSYA HATASI:", db_error)
            db_file = None

        segment_path = "static/veriler/musteri_kumeleme_sonuclari.xlsx"
        if kume_dosyasi is not None and kume_dosyasi.filename:
            segment_path = os.path.join("uploads", f"{file_id}_{kume_dosyasi.filename}")
            with open(segment_path, "wb") as buffer:
                shutil.copyfileobj(kume_dosyasi.file, buffer)

        data_obj = VRPData(filepath=veri_path, segment_filepath=segment_path)
        solver = ALNSSetPartitioning(data=data_obj, seed=42)
        sonuc = solver.solve(iterations=iterations)

        try:
            crud.create_routing_result(
                db=db, total_cost=float(sonuc["total_cost"]), iterations=iterations,
                routes=sonuc["routes"],
                user_id=current_user.id if current_user else None,
                uploaded_file_id=db_file.id if db_file else None
            )
        except Exception as db_error:
            print("DB ROTALAMA HATASI:", db_error)

        return {
            "message": "Rotalama başarıyla tamamlandı.",
            "dosya_adi": veri_dosyasi.filename,
            "iterasyon": iterations,
            "rota": sonuc["routes"],
            "maliyet": float(sonuc["total_cost"]),
            "download_excel": "/download/matheuristic_sonuc.xlsx"
        }

    except Exception as e:
        print("GENEL HATA:", e)
        raise HTTPException(status_code=500, detail=f"Kendi verileriniz ile rotalama sırasında hata oluştu: {str(e)}")


# =========================================================
# SEGMENTASYON
# =========================================================

@app.post("/segmentasyon/tez-verisi")
async def tez_verisi_segmentasyon():
    try:
        import pandas as pd
        profil_df = pd.read_excel("outputs/musteri_kumeleme_sonuclari.xlsx")
        cluster_ozet = pd.read_excel("outputs/cluster_ozet.xlsx")
        return {
            "success": True,
            "customers": profil_df.to_dict(orient="records"),
            "cluster_summary":json.loads(cluster_ozet.replace([float('inf'), float('-inf')], 0).fillna(0).to_json(orient="records", force_ascii=False)),
            "plot_url": "/static/kumeleme_sonuclari/cluster_plot.png",
            "silhouette_plot_url": "/static/kumeleme_sonuclari/silhouette_plot.png",
            "silhouette_score": 0.61,
            "best_k": 2,
            "excel_1": "/download/musteri_kumeleme_sonuclari.xlsx",
            "excel_2": "/download/cluster_ozet.xlsx"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tez segmentasyon verisi okunamadı: {str(e)}")


@app.post("/segmentasyon/kendi-verim")
async def kendi_verim_ile_segmentasyon(
    data: Annotated[UploadFile, File()],
    model_file: Optional[UploadFile] = File(default=None),
    scaler_file: Optional[UploadFile] = File(default=None),
    n_clusters: int = Form(default=3),
    window_size: int = Form(default=30),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    try:
        if not data.filename.endswith((".csv", ".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="Lütfen CSV veya Excel dosyası yükleyin.")

        file_id = str(uuid.uuid4())
        upload_path = os.path.join("uploads", f"{file_id}_{data.filename}")
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(data.file, buffer)

        model_path = "lstm_talep_model.h5"
        scaler_path = "scaler.pkl"

        if model_file is not None and model_file.filename:
            model_path = os.path.join("uploads", f"{file_id}_{model_file.filename}")
            with open(model_path, "wb") as buffer:
                shutil.copyfileobj(model_file.file, buffer)

        if scaler_file is not None and scaler_file.filename:
            scaler_path = os.path.join("uploads", f"{file_id}_{scaler_file.filename}")
            with open(scaler_path, "wb") as buffer:
                shutil.copyfileobj(scaler_file.file, buffer)

        (profil_df, cluster_ozet, kmeans, profile_scaler,
         plot_path, silhouette_plot_path, sil_score, best_k) = kumeleme_pipeline(
            data_path=upload_path, model_path=model_path, scaler_path=scaler_path,
            window_size=window_size, test_ratio=0.10, n_clusters=n_clusters,
            output_dir="outputs", plot_dir="static/plots"
        )

        cluster_plot_b64 = None
        silhouette_plot_b64 = None

        try:
            import base64 as _b64
            import io as _io

            db_file = crud.create_uploaded_file(
                db=db, filename=data.filename, saved_path=upload_path,
                file_type="segmentasyon", file_size=os.path.getsize(upload_path),
                user_id=current_user.id if current_user else None
            )

            def _path_to_b64(path):
                try:
                    with open(path, "rb") as f:
                        return _b64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    return None

            cluster_plot_b64 = _path_to_b64(plot_path)
            silhouette_plot_b64 = _path_to_b64(silhouette_plot_path)

            excel_buf = _io.BytesIO()
            profil_df.sort_values("Store").to_excel(excel_buf, index=False)
            excel_data = excel_buf.getvalue()

            crud.create_segmentation_result(
                db=db, n_clusters=n_clusters, best_k=int(best_k),
                silhouette_score=float(sil_score), window_size=window_size,
                cluster_plot_b64=cluster_plot_b64, silhouette_plot_b64=silhouette_plot_b64,
                excel_data=excel_data, cluster_plot_path=None, silhouette_plot_path=None,
                excel_output_path=None,
                cluster_summary_json=json.dumps(cluster_ozet.head(20).to_dict(orient="records"), ensure_ascii=False),
                user_id=current_user.id if current_user else None,
                uploaded_file_id=db_file.id if db_file else None
            )
        except Exception as db_err:
            print("DB SEG KAYIT HATASI:", db_err)

        return {
            "message": "Yüklenen veri ile kümeleme tamamlandı.",
            "silhouette_score": float(sil_score),
            "best_k": int(best_k),
            "plot_url": "data:image/png;base64," + (cluster_plot_b64 or ""),
            "silhouette_plot_url": "data:image/png;base64," + (silhouette_plot_b64 or ""),
            "excel_1": None,
            "excel_2": None,
            "customers": profil_df.head(100).to_dict(orient="records"),
            "cluster_summary":json.loads(cluster_ozet.head(20).replace([float('inf'), float('-inf')], 0).fillna(0).to_json(orient="records", force_ascii=False))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yüklenen veri ile kümeleme sırasında hata oluştu: {str(e)}")


# =========================================================
# LSTM — BACKGROUND TASK
# =========================================================

@app.post("/segmentasyon/lstm-egit")
async def lstm_egit_endpoint(
    data: Annotated[UploadFile, File()],
    epochs: int = Form(default=20),
    window_size: int = Form(default=30),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if not data.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Lütfen CSV veya Excel dosyası yükleyin.")

    file_id = str(uuid.uuid4())
    upload_path = os.path.join("uploads", f"{file_id}_{data.filename}")
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(data.file, buffer)

    task_id = str(uuid.uuid4())
    _lstm_tasks[task_id] = {"durum": "basliyor", "sonuc": None, "hata": None}

    def _egit():
        try:
            _lstm_tasks[task_id]["durum"] = "egitiliyor"
            sonuc = lstm_model_egit(
                data=upload_path,
                window_size=window_size,
                epochs=epochs,
                output_dir="outputs",
                plot_dir="static/plots"
            )
            try:
                db_file = crud.create_uploaded_file(
                    db=db, filename=data.filename, saved_path=upload_path,
                    file_type="lstm", file_size=os.path.getsize(upload_path),
                    user_id=current_user.id if current_user else None
                )
                crud.create_lstm_result(
                    db=db, epochs=epochs, window_size=window_size,
                    mae=sonuc["mae"], rmse=sonuc["rmse"], mape=sonuc["mape"],
                    final_loss=sonuc["final_loss"], final_val_loss=sonuc["final_val_loss"],
                    epoch_count=sonuc["epoch_count"],
                    loss_plot_b64=sonuc.get("loss_plot_b64"),
                    prediction_plot_b64=sonuc.get("prediction_plot_b64"),
                    model_data=sonuc.get("model_data"),
                    scaler_data=sonuc.get("scaler_data"),
                    loss_plot_path=None, prediction_plot_path=None,
                    model_path=None, scaler_path=None,
                    user_id=current_user.id if current_user else None,
                    uploaded_file_id=db_file.id if db_file else None
                )
            except Exception as db_err:
                print("DB LSTM KAYIT HATASI:", db_err)

            _lstm_tasks[task_id]["durum"] = "tamamlandi"
            _lstm_tasks[task_id]["sonuc"] = {
                "mae": sonuc["mae"],
                "rmse": sonuc["rmse"],
                "mape": sonuc["mape"],
                "epoch_count": sonuc["epoch_count"],
                "final_loss": sonuc["final_loss"],
                "final_val_loss": sonuc["final_val_loss"],
                "loss_plot_url": "data:image/png;base64," + sonuc.get("loss_plot_b64", ""),
                "prediction_plot_url": "data:image/png;base64," + sonuc.get("prediction_plot_b64", ""),
                "model_download": None,
                "scaler_download": None
            }
        except Exception as e:
            _lstm_tasks[task_id]["durum"] = "hata"
            _lstm_tasks[task_id]["hata"] = str(e)

    t = threading.Thread(target=_egit, daemon=True)
    t.start()

    return {"task_id": task_id}


@app.get("/segmentasyon/lstm-durum/{task_id}")
async def lstm_durum(task_id: str, current_user=Depends(get_current_user)):
    gorev = _lstm_tasks.get(task_id)
    if not gorev:
        raise HTTPException(status_code=404, detail="Görev bulunamadı.")
    return gorev


@app.delete("/segmentasyon/lstm-temizle/{task_id}")
async def lstm_temizle(task_id: str, current_user=Depends(get_current_user)):
    _lstm_tasks.pop(task_id, None)
    return {"ok": True}


# =========================================================
# DOSYA İNDİRME
# =========================================================

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")
    return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")


@app.get("/download-rotalama")
async def download_rotalama():
    path = "static/rotalama_sonuclari/matheuristic_sonuc.xlsx"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Rotalama sonucu bulunamadı.")
    return FileResponse(path=path, filename="matheuristic_sonuc.xlsx", media_type="application/octet-stream")


# =========================================================
# AUTH
# =========================================================

class UserCreate(PydanticBase):
    username: str
    email: str
    password: str


class UserOut(PydanticBase):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/token")
async def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = crud.authenticate_user(db, data.username.strip(), data.password.strip())
    if not user:
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı.")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "username": user.username}


@app.get("/auth/me")
async def me(current_user=Depends(get_current_user)):
    return {"id": current_user.id, "username": current_user.username, "email": current_user.email}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")


@app.post("/register")
async def register(
    first_name: str = Form(...),
    last_name: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    profile_image: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    if password != password_confirm:
        raise HTTPException(status_code=400, detail="Şifreler eşleşmiyor")

    existing_user = db.query(models.User).filter(
        (models.User.email == email) | (models.User.username == username)
    ).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Kullanıcı adı veya email zaten kayıtlı")

    profile_path = None
    if profile_image and profile_image.filename:
        ext = profile_image.filename.split(".")[-1].lower()
        if ext not in ["jpg", "jpeg", "png", "webp"]:
            raise HTTPException(status_code=400, detail="Desteklenmeyen resim formatı")
        filename = f"{uuid.uuid4()}.{ext}"
        os.makedirs("static/profile_images", exist_ok=True)
        profile_path = f"static/profile_images/{filename}"
        with open(profile_path, "wb") as buffer:
            shutil.copyfileobj(profile_image.file, buffer)

    user = crud.create_user(
        db=db, first_name=first_name, last_name=last_name,
        username=username, email=email, password=password,
        profile_image=profile_path
    )
    token = create_access_token({"sub": user.username})
    return {"message": "Kayıt başarılı", "access_token": token, "token_type": "bearer"}


# =========================================================
# GEÇMİŞ
# =========================================================

@app.get("/gecmis/dosyalar")
async def gecmis_dosyalar(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    dosyalar = crud.get_uploaded_files(db, user_id=current_user.id)
    return [{"id": d.id, "dosya_adi": d.filename, "tip": d.file_type, "tarih": d.uploaded_at} for d in dosyalar]


@app.get("/gecmis/rotalama")
async def gecmis_rotalama(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    sonuclar = crud.get_routing_results(db, user_id=current_user.id)
    return [{"id": s.id, "toplam_maliyet": s.total_cost, "rota_sayisi": s.route_count, "tarih": s.created_at} for s in sonuclar]


@app.get("/gecmis/lstm")
async def gecmis_lstm(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    sonuclar = crud.get_lstm_results(db, user_id=current_user.id)
    return [
        {
            "id": s.id,
            "mae": s.mae,
            "rmse": s.rmse,
            "mape": s.mape,
            "epochs": s.epochs,
            "window_size": s.window_size,
            "tarih": s.created_at,
            "model_path": s.model_path,
            "scaler_path": s.scaler_path,
            "model_download": f"/download-model/{s.id}/model",
            "scaler_download": f"/download-model/{s.id}/scaler",
        }
        for s in sonuclar
    ]


@app.get("/download-model/{lstm_id}/model")
async def download_lstm_model(lstm_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    sonuc = db.query(models.LSTMResult).filter(
        models.LSTMResult.id == lstm_id,
        models.LSTMResult.user_id == current_user.id
    ).first()
    if not sonuc or not sonuc.model_data:
        raise HTTPException(status_code=404, detail="Model dosyası bulunamadı.")
    from fastapi.responses import Response
    return Response(
        content=sonuc.model_data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=lstm_model_{lstm_id}.h5"}
    )


@app.get("/download-model/{lstm_id}/scaler")
async def download_lstm_scaler(lstm_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    sonuc = db.query(models.LSTMResult).filter(
        models.LSTMResult.id == lstm_id,
        models.LSTMResult.user_id == current_user.id
    ).first()
    if not sonuc or not sonuc.scaler_data:
        raise HTTPException(status_code=404, detail="Scaler dosyası bulunamadı.")
    from fastapi.responses import Response
    return Response(
        content=sonuc.scaler_data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=scaler_{lstm_id}.pkl"}
    )


@app.get("/gecmis/segmentasyon")
async def gecmis_segmentasyon(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    sonuclar = crud.get_segmentation_results(db, user_id=current_user.id)
    return [
        {
            "id": s.id,
            "silhouette_skoru": s.silhouette_score,
            "en_iyi_k": s.best_k,
            "n_clusters": s.n_clusters,
            "window_size": s.window_size,
            "tarih": s.created_at,
            "excel_download": f"/download-segment/{s.id}/excel",
        }
        for s in sonuclar
    ]


@app.get("/download-segment/{seg_id}/excel")
async def download_segment_excel(seg_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    sonuc = db.query(models.SegmentationResult).filter(
        models.SegmentationResult.id == seg_id,
        models.SegmentationResult.user_id == current_user.id
    ).first()
    if not sonuc or not sonuc.excel_data:
        raise HTTPException(status_code=404, detail="Kümeleme Excel dosyası bulunamadı.")
    from fastapi.responses import Response
    return Response(
        content=sonuc.excel_data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=kumeleme_{seg_id}.xlsx"}
    )
