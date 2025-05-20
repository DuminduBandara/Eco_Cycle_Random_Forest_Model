import sys
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from haversine import haversine, Unit
import polyline
import joblib
from pymongo import MongoClient
from requests.exceptions import RequestException
import os
import ast

# Base directory (current working directory)
BASE_DIR = os.getcwd()

# Helper function to get the full path for files
def get_file_path(filename):
    return os.path.join(BASE_DIR, filename)

def safe_eval_list(value):
    """Safely evaluate a string representation of a list into a Python list."""
    try:
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            result = ast.literal_eval(value)
            if not isinstance(result, list):
                return []
            return [str(item) for item in result]
        return []
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating image_urls: {value} - {e}", file=sys.stderr)
        return []

def fetch_attractions():
    """Fetch tourist attractions from Google Places and MongoDB."""
    API_KEY = "AIzaSyAOeL-fUON761cUmmht44wZNFARhVozfe0"  # Replace with your Google API key
    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    locations = [
        {"name": "Colombo", "lat": 6.9271, "lon": 79.8612},
        {"name": "Kandy", "lat": 7.2906, "lon": 80.6337},
        {"name": "Galle", "lat": 6.0535, "lon": 80.2210},
        {"name": "Sigiriya", "lat": 7.9570, "lon": 80.7603},
        {"name": "Ella", "lat": 6.8667, "lon": 81.0465},
        {"name": "Mirissa", "lat": 5.9485, "lon": 80.4718},
        {"name": "Anuradhapura", "lat": 8.3114, "lon": 80.4037},
        {"name": "Negombo", "lat": 7.2081, "lon": 79.8380},
        {"name": "Trincomalee", "lat": 8.5874, "lon": 81.2152},
        {"name": "Nuwara Eliya", "lat": 6.9497, "lon": 80.7891},
        {"name": "Polonnaruwa", "lat": 7.9403, "lon": 81.0188},
        {"name": "Dambulla", "lat": 7.8742, "lon": 80.6511},
        {"name": "Badulla", "lat": 6.9934, "lon": 81.0550},
    ]
    
    tourist_attractions = []
    for loc in locations:
        params = {
            "location": f"{loc['lat']},{loc['lon']}",
            "radius": 100000,
            "type": "tourist_attraction",
            "key": API_KEY
        }
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "OK":
                print(f"Google Places API error for {loc['name']}: {data.get('status')}", file=sys.stderr)
                continue
            for place in data.get("results", []):
                tourist_attractions.append({
                    "name": place["name"],
                    "latitude": place["geometry"]["location"]["lat"],
                    "longitude": place["geometry"]["location"]["lng"],
                    "type": "attraction",
                    "category": place.get("types", [])[0] if place.get("types") else "unknown",
                    "rating": place.get("rating", 0.0),
                    "ratings_count": place.get("user_ratings_total", 0),
                    "description": place.get("vicinity", "No description"),
                    "image_urls": [],
                    "source": "google"
                })
        except RequestException as e:
            print(f"Error fetching Google Places for {loc['name']}: {e}", file=sys.stderr)
    
    if not tourist_attractions:
        print("No tourist attractions fetched from Google Places API.", file=sys.stderr)
        return None, {"error": "No tourist attractions fetched from Google Places API"}
    
    df_google = pd.DataFrame(tourist_attractions)
    try:
        df_google.to_csv(get_file_path("tourist_attractions_google.csv"), index=False)
        print(f"Saved Google attractions to {get_file_path('tourist_attractions_google.csv')}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving tourist_attractions_google.csv: {e}", file=sys.stderr)
        return None, {"error": f"Failed to save Google attractions CSV: {str(e)}"}
    
    MONGO_URI = "mongodb+srv://vihi:vihi@itpcluster.bhmi6vu.mongodb.net/EcoCycle?retryWrites=true&w=majority"
    tourist_attractions = []
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000)
        db = client["EcoCycle"]
        collection = db["locations"]
        db_data = list(collection.find())
        client.close()
        for doc in db_data:
            try:
                tourist_attractions.append({
                    "name": doc.get("name", "Unknown Attraction"),
                    "latitude": float(doc.get("latitude", 0.0)),
                    "longitude": float(doc.get("longitude", 0.0)),
                    "type": doc.get("type", "attraction"),
                    "category": doc.get("category", "unknown"),
                    "rating": float(doc.get("rating", 0.0)),
                    "ratings_count": doc.get("ratings_count", 0),
                    "description": doc.get("description", "No description"),
                    "image_urls": doc.get("image_urls", []),
                    "source": "database"
                })
            except (TypeError, ValueError) as e:
                print(f"Error processing MongoDB document {doc.get('name', 'unknown')}: {e}", file=sys.stderr)
                continue
    except Exception as e:
        print(f"Error connecting to MongoDB Atlas: {e}, proceeding with Google data only.", file=sys.stderr)
    
    df_db = pd.DataFrame(tourist_attractions)
    if not df_db.empty:
        try:
            df_db.to_csv(get_file_path("tourist_attractions_db.csv"), index=False)
            print(f"Saved MongoDB attractions to {get_file_path('tourist_attractions_db.csv')}", file=sys.stderr)
        except Exception as e:
            print(f"Error saving tourist_attractions_db.csv: {e}", file=sys.stderr)
    else:
        print("MongoDB DataFrame is empty, proceeding with Google data only.", file=sys.stderr)
        df_db = pd.DataFrame(columns=["name", "latitude", "longitude", "type", "category", "rating", "ratings_count", "description", "image_urls", "source"])
    
    try:
        df_google = pd.read_csv(get_file_path("tourist_attractions_google.csv"))
    except (pd.errors.EmptyDataError, FileNotFoundError) as e:
        print(f"Error reading tourist_attractions_google.csv: {e}", file=sys.stderr)
        return None, {"error": "Google attractions CSV empty or not found"}
    
    df_db = df_db.reindex(columns=df_google.columns, fill_value=pd.NA)
    
    for df in [df_google, df_db]:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")
        df["name"] = df["name"].astype(str)
        df["type"] = df["type"].astype(str)
        df["category"] = df["category"].astype(str)
        df["description"] = df["description"].astype(str)
        df["source"] = df["source"].astype(str)
        df["image_urls"] = df["image_urls"].apply(lambda x: x if isinstance(x, list) else [])
        df.dropna(subset=["latitude", "longitude"], inplace=True)
    
    df_combined = pd.concat([df_google, df_db], ignore_index=True)
    
    def deduplicate_spatial(df, distance_threshold_km=0.3):
        coords = df[["latitude", "longitude"]].values
        keep_indices = []
        processed = set()
        df = df.sort_values(by="source", key=lambda x: x.map({"database": 0, "google": 1}))
        df = df.reset_index(drop=True)
        coords = df[["latitude", "longitude"]].values

        for i in range(len(df)):
            if i in processed:
                continue
            keep_indices.append(i)
            for j in range(i + 1, len(df)):
                if j in processed:
                    continue
                if df.iloc[i]["source"] == "database" and df.iloc[j]["source"] == "database":
                    if (df.iloc[i]["name"] == df.iloc[j]["name"] and
                        df.iloc[i]["latitude"] == df.iloc[j]["latitude"] and
                        df.iloc[i]["longitude"] == df.iloc[j]["longitude"]):
                        print(f"Deduplicating - Keeping: {df.iloc[i]['name']} (Source: {df.iloc[i]['source']}), Removing: {df.iloc[j]['name']} (Source: {df.iloc[j]['source']}) - Exact duplicate", file=sys.stderr)
                        processed.add(j)
                    continue
                distance = haversine(
                    (coords[i][0], coords[i][1]),
                    (coords[j][0], coords[j][1]),
                    unit=Unit.KILOMETERS
                )
                if distance < distance_threshold_km:
                    print(f"Deduplicating - Keeping: {df.iloc[i]['name']} (Source: {df.iloc[i]['source']}), Removing: {df.iloc[j]['name']} (Source: {df.iloc[j]['source']}) - Distance: {distance:.2f} km", file=sys.stderr)
                    processed.add(j)
        
        deduplicated_df = df.iloc[keep_indices].drop_duplicates(subset=["name", "latitude", "longitude"])
        return deduplicated_df
    
    df_combined = deduplicate_spatial(df_combined)
    try:
        df_combined["image_urls"] = df_combined["image_urls"].apply(lambda x: str(x) if isinstance(x, list) else str([]))
        df_combined.to_csv(get_file_path("tourist_attractions_combined.csv"), index=False)
        print(f"Saved combined attractions to {get_file_path('tourist_attractions_combined.csv')}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving tourist_attractions_combined.csv: {e}", file=sys.stderr)
        return None, {"error": f"Failed to save combined attractions CSV: {str(e)}"}
    
    return df_combined, None

def get_route_coordinates(start_lat, start_lon, end_lat, end_lon, mode="driving"):
    """Fetch route coordinates using Google Directions API."""
    API_KEY = "AIzaSyAOeL-fUON761cUmmht44wZNFARhVozfe0"
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

def main(start_lat, start_lon, end_lat, end_lon):
    print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
    
    # Fetch and combine attractions
    df_combined, error = fetch_attractions()
    if error:
        return error
    
    # Fetch route coordinates for labeling
    route_coords, error = get_route_coordinates(start_lat, start_lon, end_lat, end_lon)
    if error:
        return error
    
    # Prepare features and labels
    user_location = (start_lat, start_lon)
    df_combined["distance_to_user"] = df_combined.apply(
        lambda row: haversine(user_location, (row["latitude"], row["longitude"]), unit=Unit.KILOMETERS), axis=1
    )
    
    # Compute distance_to_segment
    route_coords_list = list(zip([coord[0] for coord in route_coords], [coord[1] for coord in route_coords]))
    df_combined["distance_to_segment"] = float("inf")
    for idx, row in df_combined.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        min_distance = float("inf")
        for route_coord in route_coords_list[::2]:
            distance = haversine(route_coord, spot_coord, unit=Unit.KILOMETERS)
            min_distance = min(min_distance, distance)
        df_combined.at[idx, "distance_to_segment"] = min_distance
    
    # Label attractions as relevant if within 2 km of the route
    df_combined["relevant"] = 0
    for idx, row in df_combined.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        for route_coord in route_coords_list[::2]:
            distance = haversine(route_coord, spot_coord, unit=Unit.KILOMETERS)
            if distance <= 2:
                df_combined.at[idx, "relevant"] = 1
                break
    
    # Prepare feature matrix and target
    X = df_combined[["latitude", "longitude", "rating", "ratings_count", "distance_to_user", "distance_to_segment"]].fillna(0)
    y = df_combined["relevant"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best parameters: {grid_search.best_params_}", file=sys.stderr)
    print(f"Accuracy: {accuracy:.2f}", file=sys.stderr)
    print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0), file=sys.stderr)
    
    # Save the model and scaler
    try:
        model_path = get_file_path("random_forest_model.pkl")
        scaler_path = get_file_path("scaler.pkl")
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to: {model_path}", file=sys.stderr)
        print(f"Scaler saved to: {scaler_path}", file=sys.stderr)
        # Verify files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} was not created.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file {scaler_path} was not created.")
    except Exception as e:
        print(f"Error saving model or scaler: {e}", file=sys.stderr)
        return {"error": f"Failed to save model or scaler: {str(e)}"}
    
    return {"message": "Training completed successfully", "accuracy": accuracy}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a tourism attraction classifier.")
    parser.add_argument("--start_lat", type=float, required=True, help="Starting latitude")
    parser.add_argument("--start_lon", type=float, required=True, help="Starting longitude")
    parser.add_argument("--end_lat", type=float, required=True, help="Ending latitude")
    parser.add_argument("--end_lon", type=float, required=True, help="Ending longitude")
    args = parser.parse_args()
    
    result = main(args.start_lat, args.start_lon, args.end_lat, args.end_lon)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    else:
        print(result["message"], file=sys.stderr)