"""
auth.py — JWT token üretimi ve doğrulaması
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

import crud
from database import get_db


# =========================================================
# JWT AYARLARI
# =========================================================

SECRET_KEY = "tez-gizli-anahtar-bunu-degistir-2024"

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8


# =========================================================
# TOKEN OKUMA
# =========================================================

oauth2_scheme = OAuth2PasswordBearer(

    tokenUrl="/auth/token",

    auto_error=False
)


# =========================================================
# TOKEN OLUŞTUR
# =========================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
):

    to_encode = data.copy()

    expire = datetime.utcnow() + (

        expires_delta or

        timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    )

    to_encode.update({

        "exp": expire

    })

    encoded_jwt = jwt.encode(

        to_encode,

        SECRET_KEY,

        algorithm=ALGORITHM
    )

    return encoded_jwt


# =========================================================
# ZORUNLU GİRİŞ
# =========================================================

def get_current_user(

    token: str = Depends(oauth2_scheme),

    db: Session = Depends(get_db)

):

    credentials_exception = HTTPException(

        status_code=status.HTTP_401_UNAUTHORIZED,

        detail="Kimlik doğrulanamadı. Lütfen giriş yapın.",

        headers={
            "WWW-Authenticate": "Bearer"
        },
    )

    # TOKEN YOK
    if token is None:

        raise credentials_exception

    try:

        payload = jwt.decode(

            token,

            SECRET_KEY,

            algorithms=[ALGORITHM]
        )

        username: str = payload.get("sub")

        if username is None:

            raise credentials_exception

    except JWTError:

        raise credentials_exception

    user = crud.get_user_by_username(

        db,

        username
    )

    if user is None:

        raise credentials_exception

    return user


# =========================================================
# OPSİYONEL GİRİŞ
# =========================================================

def get_optional_user(

    token: str = Depends(oauth2_scheme),

    db: Session = Depends(get_db)

):

    if token is None:

        return None

    try:

        payload = jwt.decode(

            token,

            SECRET_KEY,

            algorithms=[ALGORITHM]
        )

        username: str = payload.get("sub")

        if username is None:

            return None

        user = crud.get_user_by_username(

            db,

            username
        )

        return user

    except JWTError:

        return None