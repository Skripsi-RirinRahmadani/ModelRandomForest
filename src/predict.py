import pandas as pd
import joblib
import numpy as np
import os

def get_recommendations(ph, suhu, curah_hujan, elevasi):
    # 1. Load the model and encoders
    model_path = 'models/random_forest_model.joblib'
    le_kec_path = 'models/le_kecamatan.joblib'
    data_path = 'data2/processed_dataset.csv'
    
    if not os.path.exists(model_path) or not os.path.exists(le_kec_path) or not os.path.exists(data_path):
        raise FileNotFoundError("Model, Encoders, or Processed Dataset not found. Please train the model first.")
        
    model = joblib.load(model_path)
    le_kecamatan = joblib.load(le_kec_path)
    df = pd.read_csv(data_path)
    
    # 2. Prepare input for prediction
    input_data = pd.DataFrame([[ph, suhu, curah_hujan, elevasi]], 
                              columns=['pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 'Elevasi_mdpl'])
    
    # 3. Get probabilities for all Kecamatans from Random Forest
    probabilities = model.predict_proba(input_data)[0]
    kec_names = le_kecamatan.classes_
    prob_map = dict(zip(kec_names, probabilities))
    
    # Identify the top location based on features
    best_kec = kec_names[np.argmax(probabilities)]
    top_confidence = np.max(probabilities)
    
    # 4. Calculate variety scores based on location probabilities and similarity
    feature_cols = ['pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 'Elevasi_mdpl']
    
    # Min-Max Normalization reference
    df_min = df[feature_cols].min()
    df_max = df[feature_cols].max()
    df_range = (df_max - df_min).replace(0, 1) # Avoid division by zero
    
    # Normalize current input
    normalized_input = (np.array([ph, suhu, curah_hujan, elevasi]) - df_min.values) / df_range.values
    
    # Group the dataset to get average environment parameters per variety
    variety_data = df.groupby(['Nama_Tanaman', 'Nama_Varietas'])[feature_cols].mean().reset_index()
    
    # Get all Kecamatans associated with each variety
    variety_kecamatans = df.groupby(['Nama_Tanaman', 'Nama_Varietas'])['Kecamatan'].unique().reset_index()
    variety_data = variety_data.merge(variety_kecamatans, on=['Nama_Tanaman', 'Nama_Varietas'])
    
    def calculate_final_score(row):
        # A. Location Score (sum of model probabilities for Kecamatan where this variety is grown)
        location_score = sum(prob_map.get(k, 0) for k in row['Kecamatan'])
        
        # B. Environmental Similarity (normalized Euclidean distance)
        variety_features = (row[feature_cols].values - df_min.values) / df_range.values
        dist = np.linalg.norm(normalized_input - variety_features)
        similarity_score = 1 / (1 + dist)
        
        # C. Blended Score (60% Location model score, 40% Similarity score)
        # Cap at 0.98 for realistic percentage representation
        final_score = (location_score * 0.6 + similarity_score * 0.4) * 0.98
        return final_score
    
    variety_data['Score'] = variety_data.apply(calculate_final_score, axis=1)
    
    # 5. For each plant, recommend the variety with the highest score
    best_recommendations = variety_data.sort_values('Score', ascending=False).groupby('Nama_Tanaman').head(1).reset_index()
    
    # Convert to dictionary {Plant: (Variety, Score)}
    results = {}
    for _, row in best_recommendations.iterrows():
        results[row['Nama_Tanaman']] = (row['Nama_Varietas'], row['Score'])
        
        # Cap score for printing
    return best_kec, results, top_confidence
 
if __name__ == "__main__":
    print("--- Sistem Rekomendasi Varietas Tanaman Hortikultura Aceh Utara (Random Forest) ---")
    try:
        print("\nInput Data Lingkungan:")
        ph = float(input("pH Tanah (contoh 6.46): ") or 6.46)
        suhu = float(input("Suhu C (contoh 24.2): ") or 24.2)
        hujan = float(input("Curah Hujan mm (contoh 1763): ") or 1763)
        elevasi = float(input("Elevasi mdpl (contoh 255): ") or 255)
        
        kec, recs, conf = get_recommendations(ph, suhu, hujan, elevasi)
        
        print("\n" + "="*70)
        print(f"Hasil Analisis Geografis:")
        print(f"- Teridentifikasi Kecamatan Terdekat : {kec}")
        print(f"- Tingkat Keyakinan Geografis Model  : {conf * 100:.2f}%")
        print("="*70)
        print("\nRekomendasi Varietas Terbaik per Komoditas:")
        for plant, (variety, score) in sorted(recs.items()):
            print(f"- {plant:<18}: Varietas {variety:<18} (Kecocokan: {score * 100:.2f}%)")
        print("="*70)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Pastikan input benar dan model sudah dilatih (python src/train_model.py).")
