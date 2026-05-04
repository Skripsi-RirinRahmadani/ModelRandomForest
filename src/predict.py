import pandas as pd
import joblib
import numpy as np

def get_recommendations(ph, suhu, curah_hujan, elevasi, air, matahari):
    # 1. Load the model and encoders
    model = joblib.load('models/random_forest_model.joblib')
    le_kecamatan = joblib.load('models/le_kecamatan.joblib')
    
    # 2. Prepare input for prediction
    # Feature order: pH_Tanah, Suhu_C, Curah_Hujan_mm, Elevasi_mdpl, Ketersediaan_Air, Intensitas_Matahari_jam
    # Map 'air' if it's string
    if isinstance(air, str):
        water_mapping = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
        air = water_mapping.get(air, 0)
        
    input_data = np.array([[ph, suhu, curah_hujan, elevasi, air, matahari]])
    
    # 3. Predict Kecamatan
    kec_encoded = model.predict(input_data)[0]
    kec_name = le_kecamatan.inverse_transform([kec_encoded])[0]
    
    # 4. Lookup varieties for this Kecamatan in the original dataset
    # We use the processed dataset to find the varieties
    df = pd.read_csv('data/processed_dataset.csv')
    recommendations = df[df['Kecamatan'] == kec_name]
    
    # Group by plant and pick the top variety (or just list them)
    # The image shows one variety per plant
    results = recommendations.groupby('Nama_Tanaman')['Nama_Varietas'].first()
    
    return kec_name, results

if __name__ == "__main__":
    print("--- Sistem Rekomendasi Varietas Hortikultura (Random Forest) ---")
    try:
        # Example inputs (you can change these or add input() prompts)
        print("\nInput Data Lingkungan:")
        ph = float(input("pH Tanah: ") or 6.46)
        suhu = float(input("Suhu (C): ") or 24.2)
        hujan = float(input("Curah Hujan (mm): ") or 1763)
        elevasi = float(input("Elevasi (mdpl): ") or 255)
        air_str = input("Ketersediaan Air (Rendah/Sedang/Tinggi): ") or "Rendah"
        matahari = float(input("Intensitas Matahari (jam): ") or 6.4)
        
        kec, recs = get_recommendations(ph, suhu, hujan, elevasi, air_str, matahari)
        
        print(f"\nLokasi Teridentifikasi: Kecamatan {kec}")
        print("\nRekomendasi:")
        for plant, variety in recs.items():
            print(f"- {plant} Varietas {variety}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Pastikan input benar dan model sudah dilatih (Milestone 2).")
