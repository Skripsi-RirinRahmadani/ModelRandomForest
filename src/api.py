from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os

app = FastAPI(
    title="Horticulture Recommendation API",
    description="API untuk sistem rekomendasi varietas tanaman hortikultura menggunakan Random Forest",
    version="1.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Input Schema
class EnvironmentalInput(BaseModel):
    ph_tanah: float
    suhu_c: float
    curah_hujan_mm: float
    elevasi_mdpl: float
    ketersediaan_air: str  # Rendah, Sedang, Tinggi
    intensitas_matahari_jam: float

# Global variables to store loaded resources
MODEL = None
LE_KECAMATAN = None
DATASET = None

@app.on_event("startup")
def startup_event():
    """Memuat model dan data ke memori saat API dinyalakan."""
    global MODEL, LE_KECAMATAN, DATASET
    
    MODEL_PATH = 'models/random_forest_model.joblib'
    KECAMATAN_ENCODER_PATH = 'models/le_kecamatan.joblib'
    DATA_PATH = 'data/processed_dataset.csv'
    
    print("Loading resources into memory...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(KECAMATAN_ENCODER_PATH):
        print("ERROR: Model files not found. API might not work correctly.")
        return

    MODEL = joblib.load(MODEL_PATH)
    LE_KECAMATAN = joblib.load(KECAMATAN_ENCODER_PATH)
    DATASET = pd.read_csv(DATA_PATH)
    print("Resources loaded successfully. Ready to predict!")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to Horticulture Recommendation API. Visit /docs for documentation.",
        "model_loaded": MODEL is not None
    }

@app.post("/predict")
def predict(input_data: EnvironmentalInput):
    if MODEL is None or LE_KECAMATAN is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please check server logs.")
        
    try:
        # 1. Preprocess input (Case-insensitive handling)
        water_mapping = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
        
        # Mengubah input menjadi capitalize (Huruf pertama besar, sisanya kecil)
        formatted_air = input_data.ketersediaan_air.strip().capitalize()
        air_val = water_mapping.get(formatted_air)
        if air_val is None:
            raise HTTPException(status_code=400, detail="ketersediaan_air must be 'Rendah', 'Sedang', or 'Tinggi'")
            
        features = np.array([[
            input_data.ph_tanah,
            input_data.suhu_c,
            input_data.curah_hujan_mm,
            input_data.elevasi_mdpl,
            air_val,
            input_data.intensitas_matahari_jam
        ]])
        
        # 2. Predict Kecamatan and get confidence score
        features_df = pd.DataFrame(features, columns=[
            'pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 
            'Elevasi_mdpl', 'Ketersediaan_Air', 'Intensitas_Matahari_jam'
        ])
        
        # 2. Get probabilities for all Kecamatans
        probabilities = MODEL.predict_proba(features_df)[0]
        kec_names = LE_KECAMATAN.classes_
        prob_map = dict(zip(kec_names, probabilities))
        
        # Identified location (highest probability)
        best_kec = kec_names[np.argmax(probabilities)]
        top_confidence = np.max(probabilities)
        
        # 3. Calculate score for each variety in the dataset
        # Define features for similarity calculation
        feature_cols = ['pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 'Elevasi_mdpl', 'Ketersediaan_Air', 'Intensitas_Matahari_jam']
        
        # Simple Min-Max Normalization to ensure fair Euclidean distance
        df_min = DATASET[feature_cols].min()
        df_max = DATASET[feature_cols].max()
        df_range = (df_max - df_min).replace(0, 1) # Avoid division by zero
        
        # Normalize current input
        current_input = np.array([
            input_data.ph_tanah, input_data.suhu_c, input_data.curah_hujan_mm,
            input_data.elevasi_mdpl, air_val, input_data.intensitas_matahari_jam
        ])
        normalized_input = (current_input - df_min.values) / df_range.values
        
        # Group by plant and variety to get their average environment and locations
        variety_data = DATASET.groupby(['Nama_Tanaman', 'Nama_Varietas'])[feature_cols].mean().reset_index()
        variety_kecamatans = DATASET.groupby(['Nama_Tanaman', 'Nama_Varietas'])['Kecamatan'].unique().reset_index()
        variety_data = variety_data.merge(variety_kecamatans, on=['Nama_Tanaman', 'Nama_Varietas'])
        
        def calculate_final_score(row):
            # A. Location Match (sum of probabilities of locations)
            location_score = sum(prob_map.get(k, 0) for k in row['Kecamatan'])
            
            # B. Environmental Similarity (distance to variety's average environment)
            variety_features = (row[feature_cols].values - df_min.values) / df_range.values
            dist = np.linalg.norm(normalized_input - variety_features)
            similarity_score = 1 / (1 + dist)
            
            # C. Blended Score (60% Location, 40% Similarity)
            # Capped at 0.98 to avoid unrealistic "perfect 100%"
            return (location_score * 0.6 + similarity_score * 0.4) * 0.98
        
        variety_data['Score'] = variety_data.apply(calculate_final_score, axis=1)
        
        # 4. For each plant, pick the best variety
        best_recs = variety_data.sort_values('Score', ascending=False).groupby('Nama_Tanaman').head(1)
        
        # 5. Format response
        results_list = []
        raw_data = {}
        for _, row in best_recs.iterrows():
            plant = row['Nama_Tanaman']
            variety = row['Nama_Varietas']
            score = row['Score']
            
            results_list.append({
                "tanaman": plant,
                "varietas": variety,
                "kecocokan": f"{score * 100:.2f}%"
            })
            raw_data[plant] = {
                "varietas": variety,
                "skor_numerik": float(score)
            }
        
        return {
            "status": "success",
            "identified_location": best_kec,
            "location_confidence": f"{top_confidence * 100:.2f}%",
            "recommendations": results_list,
            "raw_data": raw_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Menjalankan server
    uvicorn.run(app, host="0.0.0.0", port=8000)
