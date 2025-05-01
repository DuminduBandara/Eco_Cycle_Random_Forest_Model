import sys
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from haversine import haversine, Unit
import polyline
import os
import ast
import joblib
from pymongo import MongoClient
from requests.exceptions import RequestException
import argparse
import json 

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

def main(start_lat, start_lon, end_lat, end_lon):
    # Step 1: Fetch Tourist Attractions Using Google Places API
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
                    "description": place.get("vicinity", "No description"),
                    "image_urls": [],
                    "source": "google"
                })
        except RequestException as e:
            print(f"Error fetching Google Places for {loc['name']}: {e}", file=sys.stderr)
    
    if not tourist_attractions:
        print("No tourist attractions fetched from Google Places API.", file=sys.stderr)
        return {"error": "No tourist attractions fetched from Google Places API"}
    
    df_google = pd.DataFrame(tourist_attractions)
    
    # Step 2: Fetch Tourist Attractions from MongoDB Atlas
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
    if df_db.empty:
        print("MongoDB DataFrame is empty, proceeding with Google data only.", file=sys.stderr)
        df_db = pd.DataFrame(columns=["name", "latitude", "longitude", "type", "category", "rating", "description", "image_urls", "source"])
    
    # Step 3: Combine Google and MongoDB Datasets
    df_db = df_db.reindex(columns=df_google.columns, fill_value=pd.NA)
    
    for df in [df_google, df_db]:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["name"] = df["name"].astype(str)
        df["type"] = df["type"].astype(str)
        df["category"] = df["category"].astype(str)
        df["description"] = df["description"].astype(str)
        df["source"] = df["source"].astype(str)
        df["image_urls"] = df["image_urls"].apply(lambda x: x if isinstance(x, list) else [])
        df.dropna(subset=["latitude", "longitude"], inplace=True)
    
    df_combined = pd.concat([df_google, df_db], ignore_index=True)
    
    def deduplicate_spatial(df, distance_threshold_km=0.5):
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
    
    # Step 4: Fetch Route Coordinates for Labeling
    start_point = (start_lat, start_lon)
    end_point = (end_lat, end_lon)
    DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_point[0]},{start_point[1]}",
        "destination": f"{end_point[0]},{end_point[1]}",
        "mode": "driving",
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
                return {"error": f"Error fetching route: {data['status']} (tried driving and walking modes)"}
        else:
            print(f"Error fetching route: {data['status']}", file=sys.stderr)
            return {"error": f"Error fetching route: {data['status']}"}
    except RequestException as e:
        print(f"Error fetching route: {str(e)}", file=sys.stderr)
        return {"error": f"Error fetching route: {str(e)}"}
    
    # Step 5: Prepare Features and Labels
    user_location = (start_lat, start_lon)
    df_combined["distance_to_user"] = df_combined.apply(
        lambda row: haversine(user_location, (row["latitude"], row["longitude"]), unit=Unit.KILOMETERS), axis=1
    )
    
    # Label attractions as relevant if within 2 km of the route
    df_combined["relevant"] = 0
    route_coords_list = list(zip(df_combined["latitude"], df_combined["longitude"]))
    for idx, row in df_combined.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        for route_coord in route_coords_list[::5]:
            distance = haversine(route_coord, spot_coord, unit=Unit.KILOMETERS)
            if distance <= 2:
                df_combined.at[idx, "relevant"] = 1
                break
    
    X = df_combined[["latitude", "longitude", "rating", "distance_to_user"]]
    y = df_combined["relevant"]
    
    # Step 6: Train Random Forest Classifier
    unique_y_classes = np.unique(y)
    if len(unique_y_classes) < 2:
        print("Error: Only one class present in y. Cannot train classifier.", file=sys.stderr)
        return {"error": "Only one class present in target variable"}
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        print("Warning: Stratified split failed, using non-stratified split.", file=sys.stderr)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    try:
        joblib.dump(clf, get_file_path("random_forest_model.pkl"))
        print("Random Forest model saved as random_forest_model.pkl", file=sys.stderr)
    except Exception as e:
        print(f"Error saving random_forest_model.pkl: {e}", file=sys.stderr)
        return {"error": f"Failed to save Random Forest model: {str(e)}"}
    
    return {"status": "Model trained and saved successfully"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tourism classifier with start and end coordinates.")
    parser.add_argument("--start_lat", type=float, required=True, help="Starting latitude")
    parser.add_argument("--start_lon", type=float, required=True, help="Starting longitude")
    parser.add_argument("--end_lat", type=float, required=True, help="Ending latitude")
    parser.add_argument("--end_lon", type=float, required=True, help="Ending longitude")
    args = parser.parse_args()
    
    # Validate coordinates
    if not (-90 <= args.start_lat <= 90 and -180 <= args.start_lon <= 180 and
            -90 <= args.end_lat <= 90 and -180 <= args.end_lon <= 180):
        print("Error: Invalid coordinates", file=sys.stderr)
        sys.exit(1)
    
    # Run the main function
    result = main(args.start_lat, args.start_lon, args.end_lat, args.end_lon)
    
    # Print JSON result
    print(json.dumps(result, indent=2))