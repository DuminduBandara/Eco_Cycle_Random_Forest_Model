from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from predict_tourism import predict_attractions

app = FastAPI()

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

@app.post("/predict")
async def predict_tourism(coords: Coordinates):
    try:
        # Validate coordinates
        if not (-90 <= coords.start_lat <= 90 and -180 <= coords.start_lon <= 180 and
                -90 <= coords.end_lat <= 90 and -180 <= coords.end_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinate values")

        # Check if model file exists
        model_path = get_file_path("random_forest_model.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file 'random_forest_model.pkl' not found")

        # Run the predict_tourism function
        result = predict_attractions(
            coords.start_lat,
            coords.start_lon,
            coords.end_lat,
            coords.end_lon,
            model_path
        )

        # Check if result contains an error
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)