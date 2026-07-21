from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
from contextlib import asynccontextmanager

# Global variables to store loaded resources
MODEL = None
LE_KECAMATAN = None
DATASET = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Memuat model dan data ke memori saat API dinyalakan."""
    global MODEL, LE_KECAMATAN, DATASET
    
    MODEL_PATH = 'models/random_forest_model.joblib'
    KECAMATAN_ENCODER_PATH = 'models/le_kecamatan.joblib'
    DATA_PATH = 'data3/processed_dataset.csv'
    
    print("Loading resources into memory...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(KECAMATAN_ENCODER_PATH) or not os.path.exists(DATA_PATH):
        print("ERROR: Model files or dataset not found. API might not work correctly.")
    else:
        MODEL = joblib.load(MODEL_PATH)
        LE_KECAMATAN = joblib.load(KECAMATAN_ENCODER_PATH)
        DATASET = pd.read_csv(DATA_PATH)
        print("Resources loaded successfully. Ready to predict!")
    yield

app = FastAPI(
    title="Horticulture Recommendation API",
    description="API untuk sistem rekomendasi varietas tanaman hortikultura menggunakan Random Forest",
    version="1.3.0",
    lifespan=lifespan
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

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to Horticulture Recommendation API. Visit /docs for documentation.",
        "model_loaded": MODEL is not None
    }

@app.post("/predict")
def predict(input_data: EnvironmentalInput, known_kecamatan: str = None):
    if MODEL is None or LE_KECAMATAN is None or DATASET is None:
        raise HTTPException(status_code=500, detail="Model/Dataset is not loaded. Please check server startup.")

    print("\n" + "="*60)
    print("🔬 PREDIKSI VARIETAS TANAMAN HORTIKULTURA")
    print("="*60)
    print(f"📊 Input Parameter:")
    print(f"   • pH Tanah          : {input_data.ph_tanah}")
    print(f"   • Suhu (°C)         : {input_data.suhu_c}")
    print(f"   • Curah Hujan (mm)  : {input_data.curah_hujan_mm}")
    print(f"   • Elevasi (mdpl)    : {input_data.elevasi_mdpl}")
    print("-"*60)

    try:
        # 1. Preprocess input
        features = np.array([[
            input_data.ph_tanah,
            input_data.suhu_c,
            input_data.curah_hujan_mm,
            input_data.elevasi_mdpl
        ]])

        # 2. Predict Kecamatan and get confidence score
        features_df = pd.DataFrame(features, columns=[
            'pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 'Elevasi_mdpl'
        ])

        # Get probabilities for all Kecamatans
        probabilities = MODEL.predict_proba(features_df)[0]
        kec_names = LE_KECAMATAN.classes_
        prob_map = dict(zip(kec_names, probabilities))

        if known_kecamatan is not None:
            # The kecamatan is already known (user picked it explicitly) — trust it
            # instead of re-guessing via the classifier, which can misfire between
            # kecamatan with overlapping environmental profiles (e.g. Baktiya Barat
            # vs Samudera).
            best_kec = known_kecamatan
            top_confidence = prob_map.get(known_kecamatan, 1.0)
            prob_map = {k: (1.0 if k == known_kecamatan else 0.0) for k in kec_names}
        else:
            # Identified location (highest probability)
            best_kec = kec_names[np.argmax(probabilities)]
            top_confidence = np.max(probabilities)
        
        # 3. Calculate score for each variety in the dataset
        # Define features for similarity calculation
        feature_cols = ['pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 'Elevasi_mdpl']
        
        # Simple Min-Max Normalization to ensure fair Euclidean distance
        df_min = DATASET[feature_cols].min()
        df_max = DATASET[feature_cols].max()
        df_range = (df_max - df_min).replace(0, 1) # Avoid division by zero
        
        # Normalize current input
        current_input = np.array([
            input_data.ph_tanah, input_data.suhu_c, input_data.curah_hujan_mm, input_data.elevasi_mdpl
        ])
        normalized_input = (current_input - df_min.values) / df_range.values
        
        # Group by plant and variety to get their average environment and locations
        variety_data = DATASET.groupby(['Nama_Tanaman', 'Nama_Varietas'])[feature_cols].mean().reset_index()
        variety_kecamatans = DATASET.groupby(['Nama_Tanaman', 'Nama_Varietas'])['Kecamatan'].unique().reset_index()
        variety_data = variety_data.merge(variety_kecamatans, on=['Nama_Tanaman', 'Nama_Varietas'])
        
        def calculate_final_score(row):
            # A. Location Match (max of probabilities of locations)
            location_score = max(prob_map.get(k, 0) for k in row['Kecamatan'])
            
            # B. Environmental Similarity (distance to variety's average environment)
            variety_features = (row[feature_cols].values - df_min.values) / df_range.values
            dist = np.linalg.norm(normalized_input - variety_features)
            similarity_score = 1 / (1 + dist)
            
            # C. Blended Score (60% Location, 40% Similarity)
            return location_score * 0.6 + similarity_score * 0.4
        
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

        # Print hasil prediksi
        print(f"✅ Hasil Prediksi:")
        print(f"   🌍 Kecamatan Teridentifikasi : {best_kec}")
        print(f"   📍 Confidence Score          : {top_confidence * 100:.2f}%")
        print(f"\n🌱 Rekomendasi Varietas Teratas (Top 5):")
        for idx, rec in enumerate(results_list[:5], 1):
            print(f"   {idx}. {rec['tanaman']}")
            print(f"      └─ Varietas: {rec['varietas']} | Kecocokan: {rec['kecocokan']}")
        print("="*60 + "\n")

        return {
            "status": "success",
            "identified_location": best_kec,
            "location_confidence": f"{top_confidence * 100:.2f}%",
            "recommendations": results_list,
            "raw_data": raw_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kecamatan")
def get_kecamatan_list():
    if DATASET is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")
    unique_kec = sorted(DATASET['Kecamatan'].unique().tolist())
    return {
        "status": "success",
        "kecamatan": unique_kec
    }

@app.get("/kecamatan/{kecamatan_name}/recommend")
def recommend_by_kecamatan(kecamatan_name: str):
    if MODEL is None or LE_KECAMATAN is None or DATASET is None:
        raise HTTPException(status_code=500, detail="Model/Dataset not loaded.")

    print("\n" + "="*60)
    print("🗺️  REKOMENDASI BERDASARKAN KECAMATAN")
    print("="*60)
    print(f"📍 Kecamatan: {kecamatan_name}")

    kec_rows = DATASET[DATASET['Kecamatan'].str.lower() == kecamatan_name.lower()]
    if kec_rows.empty:
        raise HTTPException(status_code=404, detail=f"Kecamatan '{kecamatan_name}' not found.")

    # Use the dataset's canonical casing/spelling for the kecamatan name
    canonical_kecamatan = kec_rows['Kecamatan'].iloc[0]

    ph = float(kec_rows['pH_Tanah'].mean())
    # Fill missing suhu using median or mean if needed, but since our processed_dataset already imputed suhu, mean is fully pre-imputed!
    suhu = float(kec_rows['Suhu_C'].mean())
    curah_hujan = float(kec_rows['Curah_Hujan_mm'].mean())
    elevasi = float(kec_rows['Elevasi_mdpl'].mean())

    print(f"📊 Parameter Ekologis:")
    print(f"   • pH Tanah          : {ph:.2f}")
    print(f"   • Suhu (°C)         : {suhu:.1f}")
    print(f"   • Curah Hujan (mm)  : {curah_hujan:.0f}")
    print(f"   • Elevasi (mdpl)    : {elevasi:.0f}")
    print("-"*60)
    
    env_input = EnvironmentalInput(
        ph_tanah=ph,
        suhu_c=suhu,
        curah_hujan_mm=curah_hujan,
        elevasi_mdpl=elevasi
    )

    res = predict(env_input, known_kecamatan=canonical_kecamatan)
    res["environmental_parameters"] = {
        "ph": round(ph, 2),
        "temperature": round(suhu, 1),
        "rainfall": round(curah_hujan, 0),
        "elevation": round(elevasi, 0)
    }
    return res

if __name__ == "__main__":
    import uvicorn
    # Menjalankan server
    uvicorn.run(app, host="0.0.0.0", port=8000)
