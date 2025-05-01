import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict_tourism import predict_tourism_attractions
from azureml.core.model import Model

app = FastAPI()

class Coordinates(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

def init():
    global model
    # Load model from Azure ML registered model path
    model_path = Model.get_model_path("random_forest_model")
    model = joblib.load(model_path)

@app.post("/predict")
async def predict_tourism(coords: Coordinates):
    try:
        # Call predict_tourism_attractions with hardcoded model
        result = predict_tourism_attractions(
            start_lat=coords.start_lat,
            start_lon=coords.start_lon,
            end_lat=coords.end_lat,
            end_lon=coords.end_lon,
            model=model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))