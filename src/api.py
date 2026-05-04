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
        
        # 2. Predict Kecamatan using DataFrame with feature names to avoid UserWarning
        features_df = pd.DataFrame(features, columns=[
            'pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 
            'Elevasi_mdpl', 'Ketersediaan_Air', 'Intensitas_Matahari_jam'
        ])
        
        kec_encoded = MODEL.predict(features_df)[0]
        kec_name = LE_KECAMATAN.inverse_transform([kec_encoded])[0]
        
        # 3. Lookup varieties using pre-loaded DATASET
        recommendations = DATASET[DATASET['Kecamatan'] == kec_name]
        plant_recs = recommendations.groupby('Nama_Tanaman')['Nama_Varietas'].first().to_dict()
        
        # 4. Format response
        formatted_recs = [f"{plant} Varietas {variety}" for plant, variety in plant_recs.items()]
        
        return {
            "status": "success",
            "identified_location": kec_name,
            "recommendations": formatted_recs,
            "raw_data": plant_recs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Menjalankan server
    uvicorn.run(app, host="0.0.0.0", port=8000)
