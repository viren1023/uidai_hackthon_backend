from fastapi import FastAPI
from app.api import auth

app = FastAPI(title="UIDAI Hackathon 2026 API")

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])

@app.get('/')
def root():
    return {"status": "Online", "challenge": "UIDAI Data Hackathon 2026"}