from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from predict_tourism import predict_attractions, predict_return_route
from pymongo import MongoClient
from typing import List
import datetime

app = FastAPI()

# MongoDB setup with hardcoded URL
MONGODB_URI = "mongodb+srv://vihi:vihi@itpcluster.bhmi6vu.mongodb.net/EcoCycle?retryWrites=true&w=majority"
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["EcoCycle"]
locations_collection = db["locations"]

# Base directory (same as FastAPI app)
BASE_DIR = os.getcwd()

def get_file_path(filename):
    """Helper function to get the full path for files."""
    return os.path.join(BASE_DIR, filename)

# Pydantic model for request validation
class Coordinates(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

class CurrentLocation(BaseModel):
    current_lat: float
    current_lon: float
    start_lat: float
    start_lon: float

# Pydantic model for location data
class Location(BaseModel):
    name: str
    latitude: float
    longitude: float
    type: str
    category: str
    rating: float
    description: str
    image_urls: List[str]
    source: str

@app.post("/predict")
async def predict_tourism(coords: Coordinates):
    try:
        # Validate coordinates
        if not (-90 <= coords.start_lat <= 90 and -180 <= coords.start_lon <= 180 and
                -90 <= coords.end_lat <= 90 and -180 <= coords.end_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinate values")

        # Check if model and scaler files exist
        model_path = get_file_path("random_forest_model.pkl")
        scaler_path = get_file_path("scaler.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file 'random_forest_model.pkl' not found")
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail="Scaler file 'scaler.pkl' not found")

        # Run the predict_attractions function
        result = predict_attractions(
            coords.start_lat,
            coords.start_lon,
            coords.end_lat,
            coords.end_lon,
            model_path,
            scaler_path
        )

        # Check if result contains an error
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Ensure attractions include prediction probabilities
        for attraction in result.get("attractions", []):
            if "prediction_probability" not in attraction:
                attraction["prediction_probability"] = 0.0  # Default if missing
            if "predicted_relevant" not in attraction:
                attraction["predicted_relevant"] = 0  # Default if missing

        # Ensure metrics include training metrics
        metrics = result.get("metrics", {})
        if "training_confusion_matrix" not in metrics:
            metrics["training_confusion_matrix"] = {}
        if "training_cross_validation_scores" not in metrics:
            metrics["training_cross_validation_scores"] = {}
        if "training_accuracy" not in metrics:
            metrics["training_accuracy"] = 0.0
        result["metrics"] = metrics

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/return_route")
async def get_return_route(loc: CurrentLocation):
    try:
        # Validate coordinates
        if not (-90 <= loc.current_lat <= 90 and -180 <= loc.current_lon <= 180 and
                -90 <= loc.start_lat <= 90 and -180 <= loc.start_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinate values")

        # Check if model and scaler files exist
        model_path = get_file_path("random_forest_model.pkl")
        scaler_path = get_file_path("scaler.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file 'random_forest_model.pkl' not found")
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail="Scaler file 'scaler.pkl' not found")

        # Run the predict_return_route function
        result = predict_return_route(
            loc.current_lat,
            loc.current_lon,
            loc.start_lat,
            loc.start_lon,
            model_path,
            scaler_path
        )

        # Check if result contains an error
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Ensure attractions include prediction probabilities
        for attraction in result.get("attractions", []):
            if "prediction_probability" not in attraction:
                attraction["prediction_probability"] = 0.0  # Default if missing
            if "predicted_relevant" not in attraction:
                attraction["predicted_relevant"] = 0  # Default if missing

        # Ensure metrics include training metrics
        metrics = result.get("metrics", {})
        if "training_confusion_matrix" not in metrics:
            metrics["training_confusion_matrix"] = {}
        if "training_cross_validation_scores" not in metrics:
            metrics["training_cross_validation_scores"] = {}
        if "training_accuracy" not in metrics:
            metrics["training_accuracy"] = 0.0
        result["metrics"] = metrics

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/store_location")
async def store_location(location: Location):
    try:
        # Validate coordinates
        if not (-90 <= location.latitude <= 90 and -180 <= location.longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid latitude or longitude values")

        # Validate rating
        if not (0 <= location.rating <= 5):
            raise HTTPException(status_code=400, detail="Rating must be between 0 and 5")

        # Validate image_urls
        if not all(isinstance(url, str) and url.startswith('http') for url in location.image_urls):
            raise HTTPException(status_code=400, detail="All image_urls must be valid HTTP URLs")

        # Prepare document for MongoDB
        location_doc = {
            "name": location.name,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "type": location.type,
            "category": location.category,
            "rating": location.rating,
            "ratings_count": 0,  # Default value, as not provided in input
            "description": location.description,
            "image_urls": location.image_urls,
            "source": location.source,
            "created_at": datetime.datetime.utcnow()
        }

        # Check for duplicate location (same name and coordinates)
        existing_location = locations_collection.find_one({
            "name": location.name,
            "latitude": location.latitude,
            "longitude": location.longitude
        })
        if existing_location:
            raise HTTPException(status_code=400, detail="Location with same name and coordinates already exists")

        # Insert the document into MongoDB
        result = locations_collection.insert_one(location_doc)
        if result.inserted_id:
            return {"message": "Location stored successfully", "location_id": str(result.inserted_id)}
        else:
            raise HTTPException(status_code=500, detail="Failed to store location")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)