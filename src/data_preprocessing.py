import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Map Ketersediaan_Air to numerical values
    water_mapping = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
    print("Mapping 'Ketersediaan_Air' to numerical values...")
    df['Ketersediaan_Air'] = df['Ketersediaan_Air'].map(water_mapping)
    
    # 2. Initialize LabelEncoder for target Nama_Varietas
    print("Encoding 'Nama_Varietas'...")
    le_varietas = LabelEncoder()
    df['Nama_Varietas_Encoded'] = le_varietas.fit_transform(df['Nama_Varietas'])
    
    # 3. Initialize LabelEncoder for Nama_Tanaman (useful for filtering recommendations later)
    print("Encoding 'Nama_Tanaman'...")
    le_tanaman = LabelEncoder()
    df['Nama_Tanaman_Encoded'] = le_tanaman.fit_transform(df['Nama_Tanaman'])
    
    # Create artifacts directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save encoders for future use in prediction
    joblib.dump(le_varietas, 'models/le_varietas.joblib')
    joblib.dump(le_tanaman, 'models/le_tanaman.joblib')
    print("Encoders saved to 'models/' directory.")
    
    return df, le_varietas, le_tanaman

if __name__ == "__main__":
    data_path = 'data/dataset_training_random_forest.csv'
    processed_df, le_v, le_t = preprocess_data(data_path)
    
    # Save the processed dataframe for Milestone 2
    processed_df.to_csv('data/processed_dataset.csv', index=False)
    print("Processed dataset saved to 'data/processed_dataset.csv'.")
    
    print("\nPre-processing Summary:")
    print(f"Total rows: {len(processed_df)}")
    print(f"Unique Plants: {len(le_t.classes_)}")
    print(f"Unique Varieties: {len(le_v.classes_)}")
    print("\nSample Data (first 5 rows):")
    print(processed_df[['Nama_Tanaman', 'Nama_Varietas', 'Ketersediaan_Air']].head())
