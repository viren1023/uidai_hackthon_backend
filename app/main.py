from sqlite3 import Date
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query
from app.api import auth
from app.core.data_ingestion import process_uidai_csv
import json
from enum import Enum
from pydantic import BaseModel

class DataSource(str, Enum):
    biometric = "biometric"
    demographic = "demographic"
    enrolment = "enrollment"

class metadata(BaseModel):
    pass

class filter(BaseModel):
    todate:Date
    fromdate:Date
    distric:str
    pincode:str
    pass

class summary(BaseModel):
    todate:Date
    fromdate:Date
    state:str
    distric:str
    source:str
    pass

async def background_processing(contents: bytes, source: DataSource):
    # This runs after the user gets the "Processing started" message
    events = await process_uidai_csv(contents, source)
    # with open(f"extra/events_{source}.json", "w") as f:
    #     json.dump(events, f)
    if events:
        print(f"Finished processing {source} dataset")

app = FastAPI(title="UIDAI Hackathon 2026 API")

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])

@app.get('/')
def root():
    return {"status": "Online", "challenge": "UIDAI Data Hackathon 2026"}

@app.post("/upload")
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source: DataSource = Query(..., description="Select the type of dataset you are uploading")
):
    contents = await file.read()
    background_tasks.add_task(background_processing, contents, source)
    return {
        "message": f"File '{file.filename}' received and processing started in background.",
        "source_selected": source
    }

@app.post("/search_filter")
def search_filter(data:filter):
    print(data)
    pass

@app.post("metadata")
def metadata():
    
    pass

@app.post("summary")
def summary(data:summary):
    print(data)
    pass