import sys
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from haversine import haversine, Unit
import polyline
import folium
import os
import json
import ast
import joblib
from pymongo import MongoClient
from requests.exceptions import RequestException

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
    API_KEY = "AIzaSyCwXu_hZC6f6M1tuez0hWwxR0lpFg8rqxg"  # Replace with your Google API key
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
    except Exception as e:
        print(f"Error saving tourist_attractions_combined.csv: {e}", file=sys.stderr)
        return None, {"error": f"Failed to save combined attractions CSV: {str(e)}"}
    
    return df_combined, None

def get_route_coordinates(start_lat, start_lon, end_lat, end_lon, mode="driving"):
    """Fetch route coordinates using Google Directions API."""
    API_KEY = "AIzaSyCwXu_hZC6f6M1tuez0hWwxR0lpFg8rqxg"
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
    
    route_df = pd.DataFrame(route_coords, columns=["latitude", "longitude"])
    try:
        route_df.to_csv(get_file_path("route_coordinates.csv"), index=False)
    except Exception as e:
        print(f"Error saving route_coordinates.csv: {e}", file=sys.stderr)
        return None, {"error": f"Failed to save route coordinates CSV: {str(e)}"}
    
    return route_coords, None

def find_nearby_attractions(route_coords, df, filename="nearby_attractions.csv"):
    """Filter attractions within 2 km of the route."""
    try:
        df["image_urls"] = df["image_urls"].apply(safe_eval_list)
    except Exception as e:
        print(f"Error processing image_urls in DataFrame: {e}", file=sys.stderr)
        return None, {"error": f"Failed to process image_urls: {str(e)}"}
    
    # Validate required columns
    required_columns = ["name", "latitude", "longitude", "type", "category", "rating", "ratings_count", "description", "image_urls", "source"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = "Unknown Attraction" if col == "name" else 0.0 if col in ["latitude", "longitude", "rating", "ratings_count"] else [] if col == "image_urls" else "unknown"
    
    route_coords_list = list(zip([coord[0] for coord in route_coords], [coord[1] for coord in route_coords]))
    
    nearby_attractions = []
    for _, row in df.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        min_distance = float("inf")
        closest_route_point = None
        for route_coord in route_coords_list[::2]:  # Denser sampling for better accuracy
            distance = haversine(route_coord, spot_coord, unit=Unit.KILOMETERS)
            if distance < min_distance:
                min_distance = distance
                closest_route_point = route_coord
            if distance <= 2:
                nearby_attractions.append({
                    "name": row["name"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "type": row["type"],
                    "category": row["category"],
                    "rating": row["rating"],
                    "ratings_count": row["ratings_count"],
                    "description": row["description"],
                    "image_urls": row["image_urls"],
                    "distance_km": min_distance,
                    "closest_route_point": closest_route_point,
                    "source": row["source"],
                    "distance_to_segment": min_distance
                })
                break
    
    nearby_df = pd.DataFrame(nearby_attractions)
    if not nearby_df.empty:
        nearby_df = nearby_df.sort_values(by="distance_km")
        try:
            nearby_df["image_urls"] = nearby_df["image_urls"].apply(lambda x: str(x) if isinstance(x, list) else str([]))
            nearby_df.to_csv(get_file_path(filename), index=False)
        except Exception as e:
            print(f"Error saving {filename}: {e}", file=sys.stderr)
    
    return nearby_df, None

def predict_attractions(start_lat, start_lon, end_lat, end_lon, model_path, scaler_path="scaler.pkl"):
    print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
    generated_files = []
    
    # Step 1: Load the trained Random Forest model and scaler
    model_full_path = get_file_path(model_path)
    scaler_full_path = get_file_path(scaler_path)
    
    if not os.path.exists(model_full_path):
        return {"error": f"Model file not found at {model_full_path}. Please run the training script first."}
    if not os.path.exists(scaler_full_path):
        return {"error": f"Scaler file not found at {scaler_full_path}. Please run the training script first."}
    
    try:
        clf = joblib.load(model_full_path)
        print(f"Loaded Random Forest model from {model_full_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return {"error": f"Failed to load model: {str(e)}"}
    
    try:
        scaler = joblib.load(scaler_full_path)
        print(f"Loaded scaler from {scaler_full_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading scaler: {e}", file=sys.stderr)
        return {"error": f"Failed to load scaler: {str(e)}"}
    
    # Load training metrics
    training_metrics = {}
    try:
        with open(get_file_path("training_metrics.json"), "r") as f:
            training_metrics = json.load(f)
        print(f"Loaded training metrics from {get_file_path('training_metrics.json')}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading training_metrics.json: {e}, proceeding without training metrics.", file=sys.stderr)
    
    # Step 2: Fetch and Combine Attractions
    df_combined, error = fetch_attractions()
    if error:
        return error
    
    # Step 3: Fetch Route Coordinates
    route_coords, error = get_route_coordinates(start_lat, start_lon, end_lat, end_lon)
    if error:
        return error
    
    # Step 4: Filter Attractions Within 2 km
    nearby_df, error = find_nearby_attractions(route_coords, df_combined)
    if error:
        return error
    generated_files.append("nearby_attractions.csv")
    
    # Step 5: Use Loaded Model to Predict Relevant Attractions
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
    
    # Prepare features for prediction
    X = df_combined[["latitude", "longitude", "rating", "ratings_count", "distance_to_user", "distance_to_segment"]].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Predict using the loaded model
    df_combined["predicted_relevant"] = clf.predict(X_scaled)
    df_combined["prediction_probability"] = clf.predict_proba(X_scaled)[:, 1]  # Probability for positive class
    
    # Generate ground truth labels for metrics (within 2 km of the route)
    df_combined["relevant"] = 0
    for idx, row in df_combined.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        for route_coord in route_coords_list[::2]:
            distance = haversine(route_coord, spot_coord, unit=Unit.KILOMETERS)
            if distance <= 2:
                df_combined.at[idx, "relevant"] = 1
                break
    
    y_true = df_combined["relevant"]
    y_pred = df_combined["predicted_relevant"]
    
    # Compute accuracy and classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    classification_metrics = classification_report(y_true, y_pred, output_dict=True, labels=[0, 1], zero_division=0)
    
    try:
        df_combined["image_urls"] = df_combined["image_urls"].apply(lambda x: str(x) if isinstance(x, list) else str([]))
        df_combined.to_csv(get_file_path("tourist_attractions_predicted.csv"), index=False)
        generated_files.append("tourist_attractions_predicted.csv")
    except Exception as e:
        print(f"Error saving tourist_attractions_predicted.csv: {e}", file=sys.stderr)
        return {"error": f"Failed to save predicted attractions CSV: {str(e)}"}
    
    # Step 6: Generate Ground Truth and Detection Metrics
    ground_truth = []
    for _, row in df_combined.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        for route_coord in route_coords_list[::2]:
            distance = haversine(route_coord, spot_coord, unit=Unit.KILOMETERS)
            if distance <= 2:
                ground_truth.append({
                    "name": row["name"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "distance_km": distance
                })
                break
    
    ground_truth_df = pd.DataFrame(ground_truth)
    try:
        ground_truth_df.to_csv(get_file_path("ground_truth_route_attractions.csv"), index=False)
        generated_files.append("ground_truth_route_attractions.csv")
    except Exception as e:
        print(f"Error saving ground_truth_route_attractions.csv: {e}", file=sys.stderr)
    
    detected_names = set(nearby_df["name"]) if not nearby_df.empty else set()
    ground_truth_names = set(ground_truth_df["name"])
    
    true_positives = detected_names.intersection(ground_truth_names)
    false_positives = detected_names - ground_truth_names
    false_negatives = ground_truth_names - detected_names
    
    precision = recall = detection_accuracy = 0.0
    if len(detected_names.union(ground_truth_names)) > 0:
        precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
        recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
        detection_accuracy = (len(true_positives) / (len(true_positives) + len(false_positives) + len(false_negatives))) * 100
    else:
        print("Note: No attractions found near the route. Detection metrics set to 0.", file=sys.stderr)
    
    # Step 7: Generate HTML Map for Nearby Attractions
    CLUSTER_COLOR = 'blue'
    segment_colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'darkred',
        'darkblue', 'darkgreen', 'cadetblue', 'pink', 'lightblue'
    ]
    
    midpoint_lat = (start_lat + end_lat) / 2
    midpoint_lon = (start_lon + end_lon) / 2
    map_nearby = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=10)
    folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(map_nearby)
    
    segment_points = []
    current_segment = [route_coords[0]]
    cumulative_distance = 0
    segment_id = 0
    for i in range(1, len(route_coords)):
        dist = haversine(route_coords[i-1], route_coords[i], unit=Unit.KILOMETERS)
        cumulative_distance += dist
        current_segment.append(route_coords[i])
        if cumulative_distance >= 2:
            segment_points.append({
                "segment_id": segment_id,
                "points": current_segment,
                "midpoint": current_segment[len(current_segment)//2]
            })
            segment_id += 1
            current_segment = [route_coords[i]]
            cumulative_distance = 0
    if current_segment:
        segment_points.append({
            "segment_id": segment_id,
            "points": current_segment,
            "midpoint": current_segment[len(current_segment)//2]
        })
    
    if not nearby_df.empty:
        nearby_df['segment_id'] = -1
        for idx, row in nearby_df.iterrows():
            spot_coord = (row['latitude'], row['longitude'])
            closest_segment_id = -1
            min_distance_to_segment = float('inf')
            for segment in segment_points:
                segment_midpoint = segment['midpoint']
                distance = haversine(segment_midpoint, spot_coord, unit=Unit.KILOMETERS)
                if distance < min_distance_to_segment:
                    min_distance_to_segment = distance
                    closest_segment_id = segment['segment_id']
            nearby_df.at[idx, 'segment_id'] = closest_segment_id
        
        grouped_by_segment = nearby_df.groupby('segment_id')
        for segment_id, group in grouped_by_segment:
            if segment_id == -1:
                continue
            centroid_lat = group['latitude'].mean()
            centroid_lon = group['longitude'].mean()
            centroid = (centroid_lat, centroid_lon)
            max_distance = 0
            for _, row in group.iterrows():
                point = (row['latitude'], row['longitude'])
                distance = haversine(centroid, point, unit=Unit.KILOMETERS)
                if distance > max_distance:
                    max_distance = distance
            radius_meters = max(max_distance * 1000, 500)
            color = segment_colors[segment_id % len(segment_colors)]
            folium.Circle(
                location=[centroid_lat, centroid_lon],
                radius=radius_meters,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.2,
                popup=f"Segment {segment_id} (Centroid)"
            ).add_to(map_nearby)
            folium.Marker(
                location=[centroid_lat, centroid_lon],
                popup=f"Segment {segment_id} Centroid",
                icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">Segment {segment_id}</div>')
            ).add_to(map_nearby)
            for _, row in group.iterrows():
                image_html = ""
                if row['image_urls'] and row['source'] == "database":
                    for img_url in row['image_urls'][:1]:
                        image_html += f'<img src="{img_url}" width="100" height="100"><br>'
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"{row['name']}<br>{row['description']}<br>{image_html}Distance: {row['distance_km']:.2f} km<br>Segment: {segment_id}",
                    icon=folium.Icon(color=CLUSTER_COLOR, icon='info-sign')
                ).add_to(map_nearby)
    
    map_nearby.fit_bounds([[start_lat, start_lon], [end_lat, end_lon]])
    try:
        map_nearby.save(get_file_path("nearby_route_locations_map.html"))
        generated_files.append("nearby_route_locations_map.html")
    except Exception as e:
        print(f"Error saving nearby_route_locations_map.html: {e}", file=sys.stderr)
    
    # Merge classification predictions and probabilities into nearby_df
    if not nearby_df.empty:
        nearby_df = nearby_df.merge(
            df_combined[["name", "latitude", "longitude", "predicted_relevant", "prediction_probability"]],
            on=["name", "latitude", "longitude"],
            how="left"
        )
        nearby_df["predicted_relevant"] = nearby_df["predicted_relevant"].fillna(0).astype(int)
        nearby_df["prediction_probability"] = nearby_df["prediction_probability"].fillna(0.0)
        nearby_df["segment_id"] = nearby_df["segment_id"].fillna(-1).astype(int)
    
    # Prepare JSON output
    attractions = nearby_df.to_dict(orient="records") if not nearby_df.empty else []
    for attr in attractions:
        if pd.notnull(attr["closest_route_point"]):
            attr["closest_route_point"] = list(attr["closest_route_point"])
        else:
            attr["closest_route_point"] = None
        attr["image_urls"] = safe_eval_list(attr["image_urls"])
    
    route_coordinates = [{"lat": lat, "lon": lon} for lat, lon in route_coords]
    metrics = {
        "classifier_accuracy": accuracy,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_accuracy": detection_accuracy,
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives)
    }
    if training_metrics:
        metrics.update({
            "training_confusion_matrix": training_metrics.get("confusion_matrix", {}),
            "training_cross_validation_scores": training_metrics.get("cross_validation_scores", {}),
            "training_accuracy": training_metrics.get("accuracy", 0.0)
        })
    
    result = {
        "attractions": attractions,
        "route_coordinates": route_coordinates,
        "metrics": metrics,
        "generated_files": generated_files
    }
    
    # Save JSON output
    try:
        with open(get_file_path("model_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        generated_files.append("model_results.json")
        print("JSON output saved as model_results.json", file=sys.stderr)
    except Exception as e:
        print(f"Error saving model_results.json: {e}", file=sys.stderr)
        return {"error": f"Failed to save JSON output: {str(e)}"}
    
    return result

def predict_return_route(current_lat, current_lon, start_lat, start_lon, model_path, scaler_path="scaler.pkl"):
    print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
    generated_files = []
    
    # Step 1: Load the trained Random Forest model and scaler
    model_full_path = get_file_path(model_path)
    scaler_full_path = get_file_path(scaler_path)
    
    if not os.path.exists(model_full_path):
        return {"error": f"Model file not found at {model_full_path}. Please run the training script first."}
    if not os.path.exists(scaler_full_path):
        return {"error": f"Scaler file not found at {scaler_full_path}. Please run the training script first."}
    
    try:
        clf = joblib.load(model_full_path)
        print(f"Loaded Random Forest model from {model_full_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return {"error": f"Failed to load model: {str(e)}"}
    
    try:
        scaler = joblib.load(scaler_full_path)
        print(f"Loaded scaler from {scaler_full_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading scaler: {e}", file=sys.stderr)
        return {"error": f"Failed to load scaler: {str(e)}"}
    
    # Load training metrics
    training_metrics = {}
    try:
        with open(get_file_path("training_metrics.json"), "r") as f:
            training_metrics = json.load(f)
        print(f"Loaded training metrics from {get_file_path('training_metrics.json')}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading training_metrics.json: {e}, proceeding without training metrics.", file=sys.stderr)
    
    # Step 2: Fetch and Combine Attractions
    df_combined, error = fetch_attractions()
    if error:
        return error
    
    # Step 3: Fetch Main Route to Check Overlap (for non-redundancy)
    main_route_coords, error = get_route_coordinates(start_lat, start_lon, current_lat, current_lon)
    if error:
        return error
    
    # Step 4: Fetch Return Route with Non-Redundancy
    API_KEY = "AIzaSyCwXu_hZC6f6M1tuez0hWwxR0lpFg8rqxg"
    DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{current_lat},{current_lon}",
        "destination": f"{start_lat},{start_lon}",
        "mode": "bicycling",
        "key": API_KEY,
        "alternatives": "true"
    }
    try:
        response = requests.get(DIRECTIONS_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "OK":
            best_route = None
            min_overlap_points = float("inf")
            main_route_coords_set = set(tuple(coord) for coord in main_route_coords)
            
            for route in data["routes"]:
                encoded_polyline = route["overview_polyline"]["points"]
                route_coords = [(lat, lon) for lat, lon in polyline.decode(encoded_polyline)]
                route_coords_set = set(tuple(coord) for coord in route_coords)
                overlap_points = len(main_route_coords_set.intersection(route_coords_set))
                if overlap_points < min_overlap_points:
                    min_overlap_points = overlap_points
                    best_route = route_coords
            
            if best_route:
                return_route_coords = best_route
            else:
                encoded_polyline = data["routes"][0]["overview_polyline"]["points"]
                return_route_coords = [(lat, lon) for lat, lon in polyline.decode(encoded_polyline)]
        else:
            print(f"Error fetching return route: {data['status']}", file=sys.stderr)
            return {"error": f"Error fetching return route: {data['status']}"}
    except RequestException as e:
        print(f"Error fetching return route: {str(e)}", file=sys.stderr)
        return {"error": f"Error fetching return route: {str(e)}"}
    
    return_route_df = pd.DataFrame(return_route_coords, columns=["latitude", "longitude"])
    try:
        return_route_df.to_csv(get_file_path("return_route_coordinates.csv"), index=False)
        generated_files.append("return_route_coordinates.csv")
    except Exception as e:
        print(f"Error saving return_route_coordinates.csv: {e}", file=sys.stderr)
        return {"error": f"Failed to save return route coordinates CSV: {str(e)}"}
    
    # Step 5: Filter Attractions Within 2 km of Return Route
    nearby_return_df, error = find_nearby_attractions(return_route_coords, df_combined, "nearby_return_attractions.csv")
    if error:
        return error
    generated_files.append("nearby_return_attractions.csv")
    
    # Step 6: Predict Relevance for Return Route Attractions
    user_location = (current_lat, current_lon)
    df_combined["distance_to_user"] = df_combined.apply(
        lambda row: haversine(user_location, (row["latitude"], row["longitude"]), unit=Unit.KILOMETERS), axis=1
    )
    
    # Compute distance_to_segment for return route
    route_coords_list = list(zip([coord[0] for coord in return_route_coords], [coord[1] for coord in return_route_coords]))
    df_combined["distance_to_segment"] = float("inf")
    for idx, row in df_combined.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        min_distance = float("inf")
        for route_coord in route_coords_list[::2]:
            distance = haversine(route_coord, spot_coord, unit=Unit.KILOMETERS)
            min_distance = min(min_distance, distance)
        df_combined.at[idx, "distance_to_segment"] = min_distance
    
    # Prepare features for prediction
    X = df_combined[["latitude", "longitude", "rating", "ratings_count", "distance_to_user", "distance_to_segment"]].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Predict using the loaded model
    df_combined["predicted_relevant"] = clf.predict(X_scaled)
    df_combined["prediction_probability"] = clf.predict_proba(X_scaled)[:, 1]  # Probability for positive class
    
    # Step 7: Generate HTML Map for Return Route
    CLUSTER_COLOR = 'blue'
    segment_colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'darkred',
        'darkblue', 'darkgreen', 'cadetblue', 'pink', 'lightblue'
    ]
    
    midpoint_lat = (current_lat + start_lat) / 2
    midpoint_lon = (current_lon + start_lon) / 2
    map_return = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=10)
    folium.PolyLine(return_route_coords, color="green", weight=2.5, opacity=1).add_to(map_return)
    
    segment_points = []
    current_segment = [return_route_coords[0]]
    cumulative_distance = 0
    segment_id = 0
    for i in range(1, len(return_route_coords)):
        dist = haversine(return_route_coords[i-1], return_route_coords[i], unit=Unit.KILOMETERS)
        cumulative_distance += dist
        current_segment.append(return_route_coords[i])
        if cumulative_distance >= 2:
            segment_points.append({
                "segment_id": segment_id,
                "points": current_segment,
                "midpoint": current_segment[len(current_segment)//2]
            })
            segment_id += 1
            current_segment = [return_route_coords[i]]
            cumulative_distance = 0
    if current_segment:
        segment_points.append({
            "segment_id": segment_id,
            "points": current_segment,
            "midpoint": current_segment[len(current_segment)//2]
        })
    
    if not nearby_return_df.empty:
        nearby_return_df['segment_id'] = -1
        for idx, row in nearby_return_df.iterrows():
            spot_coord = (row['latitude'], row['longitude'])
            closest_segment_id = -1
            min_distance_to_segment = float('inf')
            for segment in segment_points:
                segment_midpoint = segment['midpoint']
                distance = haversine(segment_midpoint, spot_coord, unit=Unit.KILOMETERS)
                if distance < min_distance_to_segment:
                    min_distance_to_segment = distance
                    closest_segment_id = segment['segment_id']
            nearby_return_df.at[idx, 'segment_id'] = closest_segment_id
        
        grouped_by_segment = nearby_return_df.groupby('segment_id')
        for segment_id, group in grouped_by_segment:
            if segment_id == -1:
                continue
            centroid_lat = group['latitude'].mean()
            centroid_lon = group['longitude'].mean()
            centroid = (centroid_lat, centroid_lon)
            max_distance = 0
            for _, row in group.iterrows():
                point = (row['latitude'], row['longitude'])
                distance = haversine(centroid, point, unit=Unit.KILOMETERS)
                if distance > max_distance:
                    max_distance = distance
            radius_meters = max(max_distance * 1000, 500)
            color = segment_colors[segment_id % len(segment_colors)]
            folium.Circle(
                location=[centroid_lat, centroid_lon],
                radius=radius_meters,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.2,
                popup=f"Return Segment {segment_id} (Centroid)"
            ).add_to(map_return)
            folium.Marker(
                location=[centroid_lat, centroid_lon],
                popup=f"Return Segment {segment_id} Centroid",
                icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">Return Segment {segment_id}</div>')
            ).add_to(map_return)
            for _, row in group.iterrows():
                image_html = ""
                if row['image_urls'] and row['source'] == "database":
                    for img_url in row['image_urls'][:1]:
                        image_html += f'<img src="{img_url}" width="100" height="100"><br>'
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"{row['name']}<br>{row['description']}<br>{image_html}Distance: {row['distance_km']:.2f} km<br>Segment: {segment_id}",
                    icon=folium.Icon(color=CLUSTER_COLOR, icon='info-sign')
                ).add_to(map_return)
    
    map_return.fit_bounds([[current_lat, current_lon], [start_lat, start_lon]])
    try:
        map_return.save(get_file_path("return_route_map.html"))
        generated_files.append("return_route_map.html")
    except Exception as e:
        print(f"Error saving return_route_map.html: {e}", file=sys.stderr)
    
    # Step 8: Prepare JSON Output for Return Route
    if not nearby_return_df.empty:
        nearby_return_df = nearby_return_df.merge(
            df_combined[["name", "latitude", "longitude", "predicted_relevant", "prediction_probability"]],
            on=["name", "latitude", "longitude"],
            how="left"
        )
        nearby_return_df["predicted_relevant"] = nearby_return_df["predicted_relevant"].fillna(0).astype(int)
        nearby_return_df["prediction_probability"] = nearby_return_df["prediction_probability"].fillna(0.0)
        nearby_return_df["segment_id"] = nearby_return_df["segment_id"].fillna(-1).astype(int)
    
    return_attractions = nearby_return_df.to_dict(orient="records") if not nearby_return_df.empty else []
    for attr in return_attractions:
        if pd.notnull(attr["closest_route_point"]):
            attr["closest_route_point"] = list(attr["closest_route_point"])
        else:
            attr["closest_route_point"] = None
        attr["image_urls"] = safe_eval_list(attr["image_urls"])
    
    return_route_coordinates = [{"lat": lat, "lon": lon} for lat, lon in return_route_coords]
    
    result = {
        "attractions": return_attractions,
        "route_coordinates": return_route_coordinates,
        "generated_files": generated_files,
        "metrics": {
            "training_confusion_matrix": training_metrics.get("confusion_matrix", {}),
            "training_cross_validation_scores": training_metrics.get("cross_validation_scores", {}),
            "training_accuracy": training_metrics.get("accuracy", 0.0)
        }
    }
    
    # Save JSON output
    try:
        with open(get_file_path("return_route_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        generated_files.append("return_route_results.json")
        print("Return route JSON output saved as return_route_results.json", file=sys.stderr)
    except Exception as e:
        print(f"Error saving return_route_results.json: {e}", file=sys.stderr)
        return {"error": f"Failed to save return route JSON output: {str(e)}"}
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict tourist attractions along a route.")
    parser.add_argument("--start_lat", type=float, required=True, help="Starting latitude")
    parser.add_argument("--start_lon", type=float, required=True, help="Starting longitude")
    parser.add_argument("--end_lat", type=float, required=True, help="Ending latitude")
    parser.add_argument("--end_lon", type=float, required=True, help="Ending longitude")
    parser.add_argument("--model_path", type=str, default="random_forest_model.pkl", help="Path to the trained model")
    parser.add_argument("--scaler_path", type=str, default="scaler.pkl", help="Path to the scaler file")
    args = parser.parse_args()
    
    result = predict_attractions(args.start_lat, args.start_lon, args.end_lat, args.end_lon, args.model_path, args.scaler_path)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    else:
        print("Prediction completed successfully.", file=sys.stderr)