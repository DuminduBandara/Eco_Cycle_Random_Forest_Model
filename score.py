import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict_tourism import predict_attractions
from azureml.core.model import Model

app = FastAPI()

class Coordinates(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

def init():
    global model
    model_path = Model.get_model_path("random_forest_model")
    model = joblib.load(model_path)

@app.post("/predict")
async def predict_tourism(coords: Coordinates):
    try:
        if not (-90 <= coords.start_lat <= 90 and -180 <= coords.start_lon <= 180 and
                -90 <= coords.end_lat <= 90 and -180 <= coords.end_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinate values")
        result = predict_attractions(
            coords.start_lat,
            coords.start_lon,
            coords.end_lat,
            coords.end_lon,
            model
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")