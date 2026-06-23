import pandas as pd
import numpy as np
import os

def generate_training_dataset():
    print("Starting dataset generation...")
    
    # 1. Load the raw data files
    varietas_path = 'data2/data_varietas_hortikultura.csv'
    ph_path = 'data2/pH_tanah_kecamatan_aceh_utara.csv'
    rainfall_path = 'data2/curah_hujan_tahunan_kecamatan_aceh_utara_2025.csv'
    elevation_path = 'data2/elevasi_kecamatan_aceh_utara.csv'
    temp_path = 'data2/suhu_tahunan_kecamatan_aceh_utara_2025.csv'
    
    # Check if files exist
    for path in [varietas_path, ph_path, rainfall_path, elevation_path, temp_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Read dataframes
    # varietas is semicolon delimited
    df_varietas = pd.read_csv(varietas_path, sep=';')
    print(f"Loaded {len(df_varietas)} varieties from {varietas_path}")
    
    df_ph = pd.read_csv(ph_path)
    df_rainfall = pd.read_csv(rainfall_path)
    df_elevation = pd.read_csv(elevation_path)
    df_temp = pd.read_csv(temp_path)
    
    print("Loaded environmental parameters for all Kecamatan.")

    # 2. Extract and clean the relevant columns from environmental files
    # We will merge them based on 'kecamatan' (cleaned of surrounding spaces and standard capitalization)
    def clean_kecamatan_name(df):
        df['kecamatan_clean'] = df['kecamatan'].str.strip()
        return df

    df_ph = clean_kecamatan_name(df_ph)
    df_rainfall = clean_kecamatan_name(df_rainfall)
    df_elevation = clean_kecamatan_name(df_elevation)
    df_temp = clean_kecamatan_name(df_temp)

    # 3. Join all environmental data into one kecamatan reference dataframe
    # Start with pH as the base
    df_env = df_ph[['kecamatan_clean', 'ph_tanah_mean']].rename(columns={'ph_tanah_mean': 'pH_Tanah'})
    
    # Merge rainfall
    df_env = df_env.merge(
        df_rainfall[['kecamatan_clean', 'curah_hujan_tahunan']].rename(columns={'curah_hujan_tahunan': 'Curah_Hujan_mm'}),
        on='kecamatan_clean',
        how='outer'
    )
    
    # Merge elevation
    df_env = df_env.merge(
        df_elevation[['kecamatan_clean', 'elevasi_mdpl']].rename(columns={'elevasi_mdpl': 'Elevasi_mdpl'}),
        on='kecamatan_clean',
        how='outer'
    )
    
    # Merge temperature (from data/suhu_tahunan_kecamatan_aceh_utara_2025.csv)
    df_env = df_env.merge(
        df_temp[['kecamatan_clean', 'suhu_tahunan_c']].rename(columns={'suhu_tahunan_c': 'Suhu_C'}),
        on='kecamatan_clean',
        how='outer'
    )
    
    # Check if there are exactly 27 kecamatan
    print(f"Merged environmental dataframe contains {len(df_env)} Kecamatan.")
    
    # Round numerical variables to match target precisions
    df_env['pH_Tanah'] = df_env['pH_Tanah'].round(2)
    # Suhu can be float, missing values kept as NaN
    df_env['Suhu_C'] = df_env['Suhu_C'].round(1)
    df_env['Curah_Hujan_mm'] = df_env['Curah_Hujan_mm'].round(1)
    df_env['Elevasi_mdpl'] = df_env['Elevasi_mdpl'].round(1)
    
    df_env = df_env.rename(columns={'kecamatan_clean': 'Kecamatan'})

    # 4. Generate the full combination of all 30 varieties * 27 kecamatan = 810 rows
    # Set seed for reproducibility
    np.random.seed(42)
    
    rows = []
    for _, var_row in df_varietas.iterrows():
        plant = var_row['Nama_Tanaman']
        variety = var_row['Nama_Varietas']
        
        for _, env_row in df_env.iterrows():
            # Add very small Gaussian noise to features to keep them distinguishable but varied
            ph_noise = np.random.normal(0, 0.02)
            suhu_noise = np.random.normal(0, 0.05)
            hujan_noise = np.random.normal(0, 5.0)
            elev_noise = np.random.normal(0, 1.0)
            
            # Apply and round to realistic bounds
            ph = round(max(3.0, min(9.0, env_row['pH_Tanah'] + ph_noise)), 2)
            suhu = round(env_row['Suhu_C'] + suhu_noise, 1) if not pd.isna(env_row['Suhu_C']) else np.nan
            hujan = round(max(0.0, env_row['Curah_Hujan_mm'] + hujan_noise), 1)
            elev = round(max(0.0, env_row['Elevasi_mdpl'] + elev_noise), 1)
            
            rows.append({
                'Nama_Tanaman': plant,
                'Kecamatan': env_row['Kecamatan'],
                'pH_Tanah': ph,
                'Suhu_C': suhu,
                'Curah_Hujan_mm': hujan,
                'Elevasi_mdpl': elev,
                'Nama_Varietas': variety
            })
            
    df_generated = pd.DataFrame(rows)
    print(f"Generated training dataset has {len(df_generated)} rows.")
    
    # Save the generated dataset
    output_path = 'data2/dataset_training_random_forest_generated.csv'
    os.makedirs('data2', exist_ok=True)
    df_generated.to_csv(output_path, index=False)
    print(f"Training dataset successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_training_dataset()
