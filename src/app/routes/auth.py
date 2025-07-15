from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.app.db.session import SessionLocal
from src.app.deps import get_db
from src.app.models.user import User
from src.app.schemas.user import UserRegister, UserLogin, TokenResponse
from src.app.core.security import hash_password, verify_password
from src.app.core.jwt import create_access_token
from datetime import timedelta
import os

router = APIRouter()
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", 1440))
ENV_SECRET_KEY = os.getenv("SECRET_KEY")

@router.post("/register", response_model=TokenResponse)
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    if user.secret_key != ENV_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid secret key")

    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed = hash_password(user.password)
    new_user = User(username=user.username, password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)

@router.post("/login", response_model=TokenResponse)
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)

@router.post("/refresh", response_model=TokenResponse)
def refresh_token(token: str):
    from jose import JWTError
    from src.app.core.jwt import decode_token

    try:
        payload = decode_token(token)
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=403, detail="Invalid token")

        access_token = create_access_token(
            data={"sub": username},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        refresh_token = create_access_token(
            data={"sub": username},
            expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
        )

        return TokenResponse(access_token=access_token, refresh_token=refresh_token)
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid refresh token")
