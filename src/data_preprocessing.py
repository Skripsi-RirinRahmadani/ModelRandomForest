import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Handle Missing Values in Suhu_C
    missing_suhu_count = df['Suhu_C'].isnull().sum()
    if missing_suhu_count > 0:
        suhu_median = df['Suhu_C'].median()
        print(f"Handling missing values: Imputing {missing_suhu_count} missing values in 'Suhu_C' with median {suhu_median:.2f}°C...")
        df['Suhu_C'] = df['Suhu_C'].fillna(suhu_median)
    else:
        print("No missing values found in 'Suhu_C'.")
        
    # 2. (REMOVED) Map Ketersediaan_Air to numerical values (Ordinal Encoding)
    
    # 3. Initialize LabelEncoder for target Nama_Varietas
    print("Encoding target 'Nama_Varietas'...")
    le_varietas = LabelEncoder()
    df['Nama_Varietas_Encoded'] = le_varietas.fit_transform(df['Nama_Varietas'])
    
    # 4. Initialize LabelEncoder for Nama_Tanaman
    print("Encoding feature 'Nama_Tanaman'...")
    le_tanaman = LabelEncoder()
    df['Nama_Tanaman_Encoded'] = le_tanaman.fit_transform(df['Nama_Tanaman'])
    
    # 5. Initialize LabelEncoder for Kecamatan
    print("Encoding feature 'Kecamatan'...")
    le_kecamatan = LabelEncoder()
    df['Kecamatan_Encoded'] = le_kecamatan.fit_transform(df['Kecamatan'])
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save encoders for future use in prediction/API
    joblib.dump(le_varietas, 'models/le_varietas.joblib')
    joblib.dump(le_tanaman, 'models/le_tanaman.joblib')
    joblib.dump(le_kecamatan, 'models/le_kecamatan.joblib')
    print("Encoders saved successfully to 'models/' directory.")
    
    return df, le_varietas, le_tanaman, le_kecamatan

if __name__ == "__main__":
    data_path = 'data2/dataset_training_random_forest_generated.csv'
    processed_df, le_v, le_t, le_k = preprocess_data(data_path)
    
    # Save the processed dataframe
    os.makedirs('data2', exist_ok=True)
    processed_df.to_csv('data2/processed_dataset.csv', index=False)
    print("Processed dataset saved to 'data2/processed_dataset.csv'.")
    
    print("\nPre-processing Summary:")
    print(f"Total rows processed: {len(processed_df)}")
    print(f"Unique Plants encoded: {len(le_t.classes_)} ({', '.join(le_t.classes_)})")
    print(f"Unique Districts (Kecamatan) encoded: {len(le_k.classes_)}")
    print(f"Unique Varieties (Target Class) encoded: {len(le_v.classes_)}")
    print("\nSample Data (first 5 rows):")
    print(processed_df[['Nama_Tanaman', 'Nama_Tanaman_Encoded', 'Nama_Varietas', 'Nama_Varietas_Encoded', 'Suhu_C']].head())
