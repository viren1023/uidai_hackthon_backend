from sqlite3 import Date
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from app.api import auth
from app.core.data_ingestion import process_uidai_csv
import json
from datetime import datetime, time
from enum import Enum
from pydantic import BaseModel
from app.utils.database import biometric_collection,demographic_collection,enrollment_collection,metadata_collection

class DataSource(str, Enum):
    biometric = "biometric"
    demographic = "demographic"
    enrolment = "enrollment"

class metadata(BaseModel):
    pass

class filter(BaseModel):
    todate:Date
    fromdate:Date
    district:str
    pincode:str
    pass

class summary(BaseModel):
    fromdate:Date
    todate:Date
    state:str
    district:Optional[str] = None
    source:str
    limit:int=10
    pass

class total_details(BaseModel):
    fromdate:Date
    todate:Date
    state:str
    pass

class district_details(BaseModel):
    fromdate:Date
    todate:Date
    state:str
    district:str
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

@app.get("/metadetails")
async def metadetails():
    try:
        result=await metadata_collection.find().to_list()
        return (f"{result}")
    except Exception as e:
        print(e)
        return e
    pass


@app.post("/summary")
async def summary(data:summary):

    try:
        # print("hello")
        from_date = datetime.combine(data.fromdate, time.min)
        to_date = datetime.combine(data.todate, time.min)
        if data.source == "biometric":
            collection=biometric_collection
            pass
        if data.source == "demographic":
            collection=demographic_collection
            pass
        if data.source == "enrollment":
            collection=enrollment_collection
            pass
        query={
            "state": f"{data.state}",
            "date": {
                "$gte": from_date,
                "$lte": to_date
            }
        }
        if data.district!=None:
            print("hello")
            query["district"] = f"{data.district.strip().title()}"
        result = await collection.find(query).to_list(length=data.limit)
        print(result)
        event={

        }
        return(f"{result}")
    except (Exception) as e:
        print(e)
        return(f"{e}")
    pass

@app.post("/state_summary")
async def state_summary(data: total_details):

    start = datetime.combine(data.fromdate, datetime.min.time())
    end = datetime.combine(data.todate, datetime.max.time())

    leaderboard_pipeline = [
        {
            "$match": {
                "state": data.state,
                "date": {"$gte": start, "$lte": end}
            }
        },
        {
            "$project": {
                "district": 1,
                "total_enrollments": {
                    "$sum": {
                        "$map": {
                            "input": { "$objectToArray": { "$ifNull": ["$mertics", {}] } },
                            "as": "m",
                            "in": { "$ifNull": ["$$m.v", 0] }
                        }
                    }
                }
            }
        },
        {
            "$group": {
                "_id": "$district",
                "total_enrollments": { "$sum": "$total_enrollments" }
            }
        },
        {
            "$project": {
                "_id": 0,
                "district": "$_id",
                "total_enrollments": 1
            }
        },
        {
            "$sort": { "total_enrollments": -1 }
        }
    ]

    cursors = [
        enrollment_collection.aggregate(leaderboard_pipeline),
        biometric_collection.aggregate(leaderboard_pipeline),
        demographic_collection.aggregate(leaderboard_pipeline)
    ]

    results = [await c.to_list(length=None) for c in cursors]
    res_enroll, res_bio, res_demo = results

    leaderboard = {}
    for result_set in results:
        for item in result_set:
            district = item["district"]
            total = item["total_enrollments"]
            leaderboard[district] = leaderboard.get(district, 0) + total

    leaderboard_list = [
        {"district": k, "total_enrollments": v}
        for k, v in sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
    ]

    total_enrollments = sum(item["total_enrollments"] for item in res_enroll) if res_enroll else 0
    total_biometric = sum(item["total_enrollments"] for item in res_bio) if res_bio else 0
    total_demographic = sum(item["total_enrollments"] for item in res_demo) if res_demo else 0

    metrics_pipeline = [
        {
            "$match": {
                "state": data.state,
                "date": {"$gte": start, "$lte": end}
            }
        },
        {
            "$project": {
                "metrics_array": { "$objectToArray": { "$ifNull": ["$mertics", {}] } }
            }
        },
        { "$unwind": "$metrics_array" },
        {
            "$group": {
                "_id": "$metrics_array.k",
                "total": { "$sum": { "$ifNull": ["$metrics_array.v", 0] } }
            }
        },
        {
            "$project": {
                "_id": 0,
                "metric": "$_id",
                "total": 1
            }
        },
        { "$sort": {"metric": 1} }
    ]

    cursor_metrics = enrollment_collection.aggregate(metrics_pipeline)
    metrics_results = await cursor_metrics.to_list(length=None)
    metrics_summary = {item["metric"]: item["total"] for item in metrics_results}



    return {
        "state": data.state,
        "start_date": data.fromdate,
        "end_date": data.todate,
        "district_leaderboard": leaderboard_list,
        "enrollment": {
            "start_date": data.fromdate,
            "end_date": data.todate,
            "total_enrollments": total_enrollments
        },
        "biometric": {
            "start_date": data.fromdate,
            "end_date": data.todate,
            "total_biometric": total_biometric
        },
        "demographic": {
            "start_date": data.fromdate,
            "end_date": data.todate,
            "total_demographic": total_demographic
        },
        "metrics_summary": metrics_summary
    }


@app.post("/district_summary")
async def district_summary(data: district_details):

    start = datetime.combine(data.fromdate, datetime.min.time())
    end = datetime.combine(data.todate, datetime.max.time())

    match_filter = {
        "date": {"$gte": start, "$lte": end}
    }

    if data.state:
        match_filter["state"] = data.state

    if data.district:
        match_filter["district"] = data.district

    metrics_pipeline = [
        { "$match": match_filter },
        {
            "$project": {
                "metrics_array": {
                    "$objectToArray": { "$ifNull": ["$mertics", {}] }
                }
            }
        },
        { "$unwind": "$metrics_array" },
        {
            "$group": {
                "_id": "$metrics_array.k",
                "total": { "$sum": "$metrics_array.v" }
            }
        }
    ]

    metrics_cursor = enrollment_collection.aggregate(metrics_pipeline)
    metrics_result = await metrics_cursor.to_list(length=None)

    metrics_summary = {
        item["_id"]: item["total"] for item in metrics_result
    }

    total_pipeline = [
        { "$match": match_filter },
        {
            "$project": {
                "total": {
                    "$sum": {
                        "$map": {
                            "input": {
                                "$objectToArray": {
                                    "$ifNull": ["$mertics", {}]
                                }
                            },
                            "as": "m",
                            "in": { "$ifNull": ["$$m.v", 0] }
                        }
                    }
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "total": { "$sum": "$total" }
            }
        }
    ]

    enrollment_cursor = enrollment_collection.aggregate(total_pipeline)
    biometric_cursor = biometric_collection.aggregate(total_pipeline)
    demographic_cursor = demographic_collection.aggregate(total_pipeline)

    enrollment_result = await enrollment_cursor.to_list(length=1)
    biometric_result = await biometric_cursor.to_list(length=1)
    demographic_result = await demographic_cursor.to_list(length=1)

    total_enrollments = enrollment_result[0]["total"] if enrollment_result else 0
    total_biometric = biometric_result[0]["total"] if biometric_result else 0
    total_demographic = demographic_result[0]["total"] if demographic_result else 0

    return {
        "state": data.state,
        "district": data.district,
        "start_date": data.fromdate,
        "end_date": data.todate,

        "metrics_summary": metrics_summary,

        "enrollment": {
            "start_date": data.fromdate,
            "end_date": data.todate,
            "total_enrollments": total_enrollments
        },
        "biometric": {
            "start_date": data.fromdate,
            "end_date": data.todate,
            "total_biometric": total_biometric
        },
        "demographic": {
            "start_date": data.fromdate,
            "end_date": data.todate,
            "total_demographic": total_demographic
        }
    }




