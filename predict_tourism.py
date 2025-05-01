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

def predict_attractions(start_lat, start_lon, end_lat, end_lon, model_path):
    generated_files = []
    
    # Step 1: Load the trained Random Forest model
    try:
        clf = joblib.load(model_path)
        print("Loaded Random Forest model from", model_path, file=sys.stderr)
    except Exception as e:
        print(f"Error loading random_forest_model.pkl: {e}", file=sys.stderr)
        return {"error": f"Failed to load Random Forest model: {str(e)}"}
    
    # Step 2: Fetch Tourist Attractions Using Google Places API
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
    try:
        df_google.to_csv(get_file_path("tourist_attractions_google.csv"), index=False)
        generated_files.append("tourist_attractions_google.csv")
    except Exception as e:
        print(f"Error saving tourist_attractions_google.csv: {e}", file=sys.stderr)
        return {"error": f"Failed to save Google attractions CSV: {str(e)}"}
    
    # Step 3: Fetch Tourist Attractions from MongoDB Atlas
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
    if not df_db.empty:
        try:
            df_db.to_csv(get_file_path("tourist_attractions_db.csv"), index=False)
            generated_files.append("tourist_attractions_db.csv")
        except Exception as e:
            print(f"Error saving tourist_attractions_db.csv: {e}", file=sys.stderr)
    else:
        print("MongoDB DataFrame is empty, proceeding with Google data only.", file=sys.stderr)
        df_db = pd.DataFrame(columns=["name", "latitude", "longitude", "type", "category", "rating", "description", "image_urls", "source"])
    
    # Step 4: Combine Google and MongoDB Datasets
    try:
        df_google = pd.read_csv(get_file_path("tourist_attractions_google.csv"))
    except (pd.errors.EmptyDataError, FileNotFoundError) as e:
        print(f"Error reading tourist_attractions_google.csv: {e}", file=sys.stderr)
        return {"error": "Google attractions CSV empty or not found"}
    
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
    try:
        df_combined["image_urls"] = df_combined["image_urls"].apply(lambda x: str(x) if isinstance(x, list) else str([]))
        df_combined.to_csv(get_file_path("tourist_attractions_combined.csv"), index=False)
        generated_files.append("tourist_attractions_combined.csv")
    except Exception as e:
        print(f"Error saving tourist_attractions_combined.csv: {e}", file=sys.stderr)
        return {"error": f"Failed to save combined attractions CSV: {str(e)}"}
    
    # Step 5: Fetch Route Coordinates
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
            params["mode": "walking"]
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
    
    route_df = pd.DataFrame(route_coords, columns=["latitude", "longitude"])
    try:
        route_df.to_csv(get_file_path("route_coordinates.csv"), index=False)
        generated_files.append("route_coordinates.csv")
    except Exception as e:
        print(f"Error saving route_coordinates.csv: {e}", file=sys.stderr)
        return {"error": f"Failed to save route coordinates CSV: {str(e)}"}
    
    # Step 6: Filter Attractions Within 2 km
    try:
        df = pd.read_csv(get_file_path("tourist_attractions_combined.csv"))
        df["image_urls"] = df["image_urls"].apply(safe_eval_list)
    except Exception as e:
        print(f"Error reading tourist_attractions_combined.csv: {e}", file=sys.stderr)
        return {"error": f"Failed to read combined attractions CSV: {str(e)}"}
    
    route_coords_list = list(zip(route_df["latitude"], route_df["longitude"]))
    
    nearby_attractions = []
    for _, row in df.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        min_distance = float("inf")
        closest_route_point = None
        for route_coord in route_coords_list[::5]:
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
                    "description": row["description"],
                    "image_urls": row["image_urls"],
                    "distance_km": min_distance,
                    "closest_route_point": closest_route_point,
                    "source": row["source"]
                })
                break
    
    nearby_df = pd.DataFrame(nearby_attractions)
    if not nearby_df.empty:
        nearby_df = nearby_df.sort_values(by="distance_km")
        try:
            nearby_df["image_urls"] = nearby_df["image_urls"].apply(lambda x: str(x) if isinstance(x, list) else str([]))
            nearby_df.to_csv(get_file_path("nearby_attractions.csv"), index=False)
            generated_files.append("nearby_attractions.csv")
        except Exception as e:
            print(f"Error saving nearby_attractions.csv: {e}", file=sys.stderr)
    
    # Step 7: Use Loaded Model to Predict Relevant Attractions
    user_location = (start_lat, start_lon)
    df["distance_to_user"] = df.apply(
        lambda row: haversine(user_location, (row["latitude"], row["longitude"]), unit=Unit.KILOMETERS), axis=1
    )
    
    # Prepare features for prediction
    X = df[["latitude", "longitude", "rating", "distance_to_user"]]
    
    # Predict using the loaded model
    df["predicted_relevant"] = clf.predict(X)
    
    # Generate ground truth labels for metrics (within 2 km of the route)
    df["relevant"] = 0
    for idx, row in df.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        for route_coord in route_coords_list[::5]:
            distance = haversine(route_coord, spot_coord, unit=Unit.KILOMETERS)
            if distance <= 2:
                df.at[idx, "relevant"] = 1
                break
    
    y_true = df["relevant"]
    y_pred = df["predicted_relevant"]
    
    # Compute accuracy and classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    classification_metrics = classification_report(y_true, y_pred, output_dict=True, labels=[0, 1], zero_division=0)
    
    try:
        df["image_urls"] = df["image_urls"].apply(lambda x: str(x) if isinstance(x, list) else str([]))
        df.to_csv(get_file_path("tourist_attractions_predicted.csv"), index=False)
        generated_files.append("tourist_attractions_predicted.csv")
    except Exception as e:
        print(f"Error saving tourist_attractions_predicted.csv: {e}", file=sys.stderr)
        return {"error": f"Failed to save predicted attractions CSV: {str(e)}"}
    
    # Step 8: Generate Ground Truth and Detection Metrics
    ground_truth = []
    for _, row in df.iterrows():
        spot_coord = (row["latitude"], row["longitude"])
        for route_coord in route_coords_list[::5]:
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
    
    # Step 9: Generate HTML Map for Nearby Attractions
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
    
    # Merge classification predictions into nearby_df
    if not nearby_df.empty:
        nearby_df = nearby_df.merge(
            df[["name", "latitude", "longitude", "predicted_relevant"]],
            on=["name", "latitude", "longitude"],
            how="left"
        )
        nearby_df["predicted_relevant"] = nearby_df["predicted_relevant"].fillna(0).astype(int)
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