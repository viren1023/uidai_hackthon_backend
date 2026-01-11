from fastapi import APIRouter, HTTPException, Depends
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from app.utils.config import settings
from app.utils.database import user_collection

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/login")
async def login(user_data: dict):
    user = await user_collection.find_one({"email": user_data.get("email")})

    if not user or not pwd_context.verify(user_data["password"], user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    payload = {
        "sub": user["email"],
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

@router.post("/register")
async def register(user_data: dict):
    pass