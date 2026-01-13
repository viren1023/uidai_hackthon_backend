from fastapi import APIRouter, HTTPException, Depends
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from app.utils.config import settings
from app.utils.database import user_collection
from pydantic import BaseModel


router = APIRouter()
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

class LoginRequest(BaseModel):
    user_name:str
    password: str

@router.post("/login")
async def login(data:LoginRequest):
    userDB = await user_collection.find_one({"user_name": data.user_name})

    if not userDB or not pwd_context.verify(data.password, userDB["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    payload = {
        "sub": userDB["user_name"],
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}
