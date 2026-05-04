import pandas as pd

df = pd.read_csv('data/processed_dataset.csv')
print("Dataset Shape:", df.shape)
print("\nUnique values in Nama_Tanaman:")
print(df['Nama_Tanaman'].unique())

print("\nValue counts for Nama_Varietas_Encoded:")
print(df['Nama_Varietas_Encoded'].value_counts())

print("\nChecking for duplicate features with different labels:")
features = [
    'pH_Tanah', 
    'Suhu_C', 
    'Curah_Hujan_mm', 
    'Elevasi_mdpl', 
    'Ketersediaan_Air', 
    'Intensitas_Matahari_jam',
    'Nama_Tanaman_Encoded'
]
duplicates = df.duplicated(subset=features, keep=False)
print(f"Number of rows with identical features: {duplicates.sum()}")

if duplicates.sum() > 0:
    print("\nSample of duplicate features with different labels:")
    print(df[duplicates].sort_values(by=features).head(10)[features + ['Nama_Varietas']])
