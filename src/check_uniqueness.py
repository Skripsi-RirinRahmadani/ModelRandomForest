import pandas as pd

df = pd.read_csv('data/processed_dataset.csv')
kecamatan_data = df[['Kecamatan', 'pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 'Elevasi_mdpl', 'Ketersediaan_Air', 'Intensitas_Matahari_jam']].drop_duplicates()

print(f"Total rows in dataset: {len(df)}")
print(f"Total unique Kecamatan: {df['Kecamatan'].nunique()}")
print(f"Total unique environmental sets: {len(kecamatan_data)}")

if len(kecamatan_data) == df['Kecamatan'].nunique():
    print("\nSUCCESS: Each Kecamatan has a unique set of environmental factors.")
else:
    print("\nWARNING: Some Kecamatans share the same environmental factors.")
