from motor.motor_asyncio import AsyncIOMotorClient
from app.utils.config import settings

mongo_url = settings.MONGO_URL
client = AsyncIOMotorClient(mongo_url)
database = client.uidai_hackathon_db
user_collection = database.get_collection("users")
biometric_collection = database.get_collection("biometric")
demographic_collection = database.get_collection("demographic")
enrollment_collection = database.get_collection("enrollment")
metadata_collection = database.get_collection("metadata")