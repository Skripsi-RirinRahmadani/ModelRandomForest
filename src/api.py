from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os

app = FastAPI(
    title="Horticulture Recommendation API",
    description="API untuk sistem rekomendasi varietas tanaman hortikultura menggunakan Random Forest",
    version="1.0.0"
)

# Input Schema
class EnvironmentalInput(BaseModel):
    ph_tanah: float
    suhu_c: float
    curah_hujan_mm: float
    elevasi_mdpl: float
    ketersediaan_air: str  # Rendah, Sedang, Tinggi
    intensitas_matahari_jam: float

# Global variables for model and data
MODEL_PATH = 'models/random_forest_model.joblib'
KECAMATAN_ENCODER_PATH = 'models/le_kecamatan.joblib'
DATA_PATH = 'data/processed_dataset.csv'

def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(KECAMATAN_ENCODER_PATH):
        raise FileNotFoundError("Model or Encoder files not found. Please train the model first.")
    
    model = joblib.load(MODEL_PATH)
    le_kecamatan = joblib.load(KECAMATAN_ENCODER_PATH)
    df = pd.read_csv(DATA_PATH)
    return model, le_kecamatan, df

@app.get("/")
def read_root():
    return {"message": "Welcome to Horticulture Recommendation API. Visit /docs for documentation."}

@app.post("/predict")
def predict(input_data: EnvironmentalInput):
    try:
        model, le_kecamatan, df = load_resources()
        
        # 1. Preprocess input
        water_mapping = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
        air_val = water_mapping.get(input_data.ketersediaan_air)
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
        
        # 2. Predict Kecamatan
        kec_encoded = model.predict(features)[0]
        kec_name = le_kecamatan.inverse_transform([kec_encoded])[0]
        
        # 3. Lookup varieties
        recommendations = df[df['Kecamatan'] == kec_name]
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
