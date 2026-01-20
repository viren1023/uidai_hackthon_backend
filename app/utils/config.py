import os
from dotenv import load_dotenv
import urllib.parse

load_dotenv()

class Settings:
    user = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASS")
    cluster = os.getenv("MONGO_CLUSTER")
    
    safe_user = urllib.parse.quote_plus(user) if user else ""
    safe_pass = urllib.parse.quote_plus(password) if password else ""
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", "fallback_super_secret_string_123")
    ALGORITHM: str = "HS256"
    MONGO_URL = f"mongodb://localhost:27017/Local"
    # MONGO_URL = f"mongodb+srv://{safe_user}:{safe_pass}@{cluster}.mjad70j.mongodb.net/?appName={cluster}"

settings = Settings()