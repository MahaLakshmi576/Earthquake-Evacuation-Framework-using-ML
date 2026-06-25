from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json
import os

app = Flask(__name__)

# Load models and label encoder
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('linear_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)
with open('soil_le.pkl', 'rb') as f:
    soil_le = pickle.load(f)

# Load metrics for dashboard
with open('model_metrics.json', 'r') as f:
    metrics = json.load(f)

# Load dataset for safe zone lookup
df = pd.read_csv('earthquake_data.csv')

def find_nearest_safe_site(user_lat, user_lng):
    # Use Haversine or simple Euclidean for nearby sites
    # Since our data is synthetic, we'll just find the closest point in the dataset
    distances = np.sqrt((df['Latitude'] - user_lat)**2 + (df['Longitude'] - user_lng)**2)
    nearest_idx = distances.idxmin()
    row = df.iloc[nearest_idx]
    return {
        'site_name': row['RecommendedSite'],
        'lat': row['SafeSiteLat'],
        'lng': row['SafeSiteLng'],
        'distance': round(distances[nearest_idx] * 111, 2) # approx km
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        magnitude = float(data['magnitude'])
        depth = float(data['depth'])
        distance = float(data['distance'])
        building_age = float(data['building_age'])
        population_density = float(data['population_density'])
        soil_type = data['soil_type']
        
        user_lat = float(data.get('latitude', 0))
        user_lng = float(data.get('longitude', 0))

        # Encode soil type
        soil_encoded = soil_le.transform([soil_type])[0]
        
        features = np.array([[magnitude, depth, distance, building_age, population_density, soil_encoded]])
        
        # Prediction
        rf_pred = rf_model.predict(features)[0]
        lr_pred = linear_model.predict(features)[0]
        
        risk_label = "High Risk Area" if rf_pred == 1 else "Safe Zone"
        
        # Find nearby safe site
        safe_site = find_nearest_safe_site(user_lat, user_lng)
        
        return render_template('index.html', 
                               prediction=risk_label, 
                               rf_acc=metrics['rf_accuracy'],
                               lr_acc=metrics['lr_accuracy'],
                               safe_site=safe_site,
                               inputs=data)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
