from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from datetime import datetime
import uvicorn

# Import your existing logic from data_loader.py
from data_loader import process_parcel_data

app = FastAPI(title="TerraDrishti Analytics Engine")

# --- Request Model ---
class AnalysisRequest(BaseModel):
    task_id: str
    coords: List[List[float]]
    end_date: str  # YYYY-MM-DD

# --- API Endpoints ---

@app.post("/analyze/summary")
async def get_parcel_summary(request: AnalysisRequest):
    """
    Triggers Earth Engine processing and returns only the land type summary 
    and crop intensity metrics.
    """
    try:
        # 1. Trigger your existing process_parcel_data logic
        # This currently handles ee.Initialize and folder creation
        result = process_parcel_data(
            run_id=request.task_id,
            coordinates=request.coords,
            end_str=request.end_date
        )

        if result is None:
            raise HTTPException(status_code=500, detail="GEE Processing failed to return data.")

        df_all, summary_dict = result

        # 2. Return ONLY the summary_dict as requested
        return {
            "task_id": request.task_id,
            "status": "success",
            "summary": summary_dict,
            "metadata": {
                "point_count": len(df_all) if df_all is not None else 0,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)