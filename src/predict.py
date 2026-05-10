import pandas as pd
import joblib
import numpy as np

def get_recommendations(ph, suhu, curah_hujan, elevasi, air, matahari):
    # 1. Load the model and encoders
    model = joblib.load('models/random_forest_model.joblib')
    le_kecamatan = joblib.load('models/le_kecamatan.joblib')
    
    # 2. Prepare input for prediction
    if isinstance(air, str):
        water_mapping = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
        air = water_mapping.get(air, 0)
        
    input_data = pd.DataFrame([[ph, suhu, curah_hujan, elevasi, air, matahari]], 
                              columns=['pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 'Elevasi_mdpl', 'Ketersediaan_Air', 'Intensitas_Matahari_jam'])
    
    # 3. Get probabilities for all Kecamatans
    probabilities = model.predict_proba(input_data)[0]
    kec_names = le_kecamatan.classes_
    prob_map = dict(zip(kec_names, probabilities))
    
    # Identify the top location for reference
    best_kec = kec_names[np.argmax(probabilities)]
    top_confidence = np.max(probabilities)
    
    # 4. Load dataset to calculate variety scores
    df = pd.read_csv('data/processed_dataset.csv')
    
    # Group the dataset to get unique (Tanaman, Varietas) and their Kecamatans
    variety_data = df.groupby(['Nama_Tanaman', 'Nama_Varietas'])['Kecamatan'].unique().reset_index()
    
    # Calculate score for each variety
    def calculate_score(kecamatans):
        # Score is the sum of probabilities of all Kecamatans that have this variety
        return sum(prob_map.get(k, 0) for k in kecamatans)
    
    variety_data['Score'] = variety_data['Kecamatan'].apply(calculate_score)
    
    # 5. For each plant, pick the variety with the highest score
    best_recommendations = variety_data.sort_values('Score', ascending=False).groupby('Nama_Tanaman').head(1).reset_index()
    
    # Convert to dictionary {Plant: (Variety, Score)}
    results = {}
    for _, row in best_recommendations.iterrows():
        results[row['Nama_Tanaman']] = (row['Nama_Varietas'], row['Score'])
    
    return best_kec, results, top_confidence

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
        
        kec, recs, conf = get_recommendations(ph, suhu, hujan, elevasi, air_str, matahari)
        
        print(f"\nLokasi Terdekat Berdasarkan Lingkungan: Kecamatan {kec}")
        print(f"Tingkat Keyakinan Lokasi: {conf * 100:.2f}%")
        print("\nRekomendasi Varietas (dengan persentase kecocokan):")
        for plant, (variety, score) in recs.items():
            print(f"- {plant}: Varietas {variety} ({score * 100:.2f}%)")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Pastikan input benar dan model sudah dilatih (Milestone 2).")
