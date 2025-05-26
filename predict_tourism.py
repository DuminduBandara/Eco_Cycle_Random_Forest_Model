import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from haversine import haversine, Unit
import json
import os
import requests
import polyline
from requests.exceptions import RequestException

# Base directory (current working directory)
BASE_DIR = os.getcwd()

# Helper function to get the full path for files
def get_file_path(filename):
    return os.path.join(BASE_DIR, filename)

def get_route_coordinates(start_lat, start_lon, end_lat, end_lon, mode="driving"):
    """Fetch route coordinates using Google Directions API."""
    API_KEY = "AIzaSyAOeL-fUON761cUmmht44wZNFARhVozfe0"  # Replace with your Google API key
    DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_lat},{start_lon}",
        "destination": f"{end_lat},{end_lon}",
        "mode": mode,
        "key": API_KEY
    }
    try:
        response = requests.get(DIRECTIONS_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "OK":
            encoded_polyline = data["routes"][0]["overview_polyline"]["points"]
            route_coords = [(lat, lon) for lat, lon in polyline.decode(encoded_polyline)]
        elif data["status"] == "ZERO_RESULTS":
            params["mode"] = "walking"
            response = requests.get(DIRECTIONS_URL, params=params)
            response.raise_for_status()
            data = response.json()
            if data["status"] == "OK":
                encoded_polyline = data["routes"][0]["overview_polyline"]["points"]
                route_coords = [(lat, lon) for lat, lon in polyline.decode(encoded_polyline)]
            else:
                print(f"Error fetching route (both driving and walking failed): {data['status']}", file=sys.stderr)
                return None, {"error": f"Error fetching route: {data['status']} (tried driving and walking modes)"}
        else:
            print(f"Error fetching route: {data['status']}", file=sys.stderr)
            return None, {"error": f"Error fetching route: {data['status']}"}
    except RequestException as e:
        print(f"Error fetching route: {str(e)}", file=sys.stderr)
        return None, {"error": f"Error fetching route: {str(e)}"}
    
    return route_coords, None

def load_model_and_scaler():
    """Load the trained model and scaler."""
    try:
        model_path = get_file_path("random_forest_model.pkl")
        scaler_path = get_file_path("scaler.pkl")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None, {"error": "Model or scaler file not found"}
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"Model loaded from: {model_path}", file=sys.stderr)
        print(f"Scaler loaded from: {scaler_path}", file=sys.stderr)
        return model, scaler, None
    except Exception as e:
        print(f"Error loading model or scaler: {e}", file=sys.stderr)
        return None, None, {"error": f"Failed to load model or scaler: {str(e)}"}

def load_model_results():
    """Load model results from model_results.json."""
    try:
        results_path = get_file_path("model_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                model_results = json.load(f)
            print(f"Model results loaded from: {results_path}", file=sys.stderr)
            return model_results, None
        else:
            print("model_results.json not found.", file=sys.stderr)
            return {}, {"warning": "Model results file not found, using empty results"}
    except Exception as e:
        print(f"Error loading model results: {e}", file=sys.stderr)
        return {}, {"error": f"Failed to load model results: {str(e)}"}

def save_model_results(results):
    """Save model results and predictions to model_results.json."""
    try:
        results_path = get_file_path("model_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Model results saved to: {results_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error saving model results: {e}", file=sys.stderr)
        return {"error": f"Failed to save model results: {str(e)}"}

def predict_attractions(start_lat, start_lon, end_lat, end_lon):
    """Predict relevant tourist attractions based on user route."""
    model, scaler, error = load_model_and_scaler()
    if error:
        return error
    
    # Load model results
    model_results, error = load_model_results()
    if error:
        print(f"Warning: {error.get('error', error.get('warning'))}", file=sys.stderr)
        model_results = {}
    
    # Load attractions data
    try:
        df = pd.read_csv(get_file_path("tourist_attractions_combined.csv"))
        print(f"Attractions loaded from: {get_file_path('tourist_attractions_combined.csv')}", file=sys.stderr)
    except (pd.errors.EmptyDataError, FileNotFoundError) as e:
        print(f"Error reading tourist_attractions_combined.csv: {e}", file=sys.stderr)
        return {"error": "Attractions CSV empty or not found"}
    
    # Fetch route coordinates for accurate distance calculation
    route_coords, error = get_route_coordinates(start_lat, start_lon, end_lat, end_lon)
    if error:
        return error
    
    # Compute distance to user
    user_location = (start_lat, start_lon)
    df["distance_to_user"] = df.apply(
        lambda row: haversine(user_location, (row["latitude"], row["longitude"]), unit=Unit.KILOMETERS),
        axis=1
    )
    
    # Compute distance to route segment
    route_coords_list = list(zip([coord[0] for coord in route_coords], [coord[1] for coord in route_coords]))
    df["distance_to_segment"] = float("inf")
    for idx, row in df.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        min_distance = float("inf")
        for route_coord in route_coords_list[::2]:
            distance = haversine(route_coord, spot_coord, unit=Unit.KILOMETERS)
            min_distance = min(min_distance, distance)
        df.at[idx, "distance_to_segment"] = min_distance
    
    # Prepare features
    X = df[["latitude", "longitude", "rating", "ratings_count", "distance_to_user", "distance_to_segment"]].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Predict relevance and probabilities
    try:
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of being relevant (class 1)
        df["relevance"] = predictions
        df["relevance_score"] = probabilities
        
        # Filter attractions with high relevance (score > 0.5)
        relevant_df = df[df["relevance_score"] > 0.5].sort_values(by="relevance_score", ascending=False)
        
        # Prepare prediction output
        attractions = []
        for _, row in relevant_df.iterrows():
            try:
                image_urls = json.loads(row["image_urls"]) if isinstance(row["image_urls"], str) and row["image_urls"] else []
            except json.JSONDecodeError:
                image_urls = []
            attractions.append({
                "name": str(row["name"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "rating": float(row["rating"]),
                "ratings_count": int(row["ratings_count"]),
                "description": str(row["description"]),
                "category": str(row["category"]),
                "image_urls": image_urls,
                "relevance_score": float(row["relevance_score"])
            })
        
        # Combine with model results
        result = {
            "message": "Predictions completed successfully",
            "attractions": attractions,
            "model_results": model_results if model_results else {"message": "No model results available"}
        }
        
        # Update model_results.json with predictions
        updated_results = model_results.copy() if model_results else {}
        updated_results["predictions"] = attractions
        error = save_model_results(updated_results)
        if error:
            print(f"Warning: Failed to update model_results.json: {error}", file=sys.stderr)
        
        return result
    
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        return {"error": f"Prediction failed: {str(e)}"}

def predict_return_route(current_lat, current_lon, start_lat, start_lon):
    """Predict attractions for both current-to-start and start-to-current routes."""
    # Load model results
    model_results = load_model_results(start_lat)
    if isinstance(model_results, tuple):
        print(f"Warning: {model_results[1].get('error', model_results[1].get('warning'))}", file=sys.stderr)
        model_results = {}
    
    # Forward route (current to start)
    forward_result = predict_attractions(current_lat, current_lon, start_lat, start_lon)
    if "error" in forward_result:
        return forward_result
    
    # Return route (start to current)
    return_result = predict_attractions(start_lat, start_lon, current_lat, current_lon)
    if "error" in return_result:
        return return_result
    
    # Combine results
    result = {
        "message": "Predictions completed successfully for both routes",
        "forward_results": forward_result["attractions"],
        "return_results": return_result["attractions"],
        "model_results": model_results if model_results else {"message": "No model results available"}
    }
    
    # Update model_results.json with both routes
    updated_results = model_results.copy() if model_results else {}
    updated_results["forward_routes"] = forward_result["attractions"]
    updated_results["return_routes"] = return_result["attractions"]
    error = save_model_results(updated_results)
    if error:
        print(f"Warning: Failed to update model_results.json: {error}", file=sys.stderr)
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict tourist attractions.")
    parser.add_argument("--start_lat", type=float, required=True, help="Starting latitude")
    parser.add_argument("--start_lon", type=float, required=True, help="Starting longitude")
    parser.add_argument("--end_lat", type=float, required=True, help="Ending latitude")
    parser.add_argument("--end_lon", type=float, required=True, help="Ending longitude")
    args = parser.parse_args()
    
    result = predict_return_route(args.start_lat, args.start_lon, args.end_lat, args.end_lon)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    else:
        print(json.dumps(result, indent=4), file=sys.stdout)