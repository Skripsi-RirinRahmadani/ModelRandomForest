import pandas as pd
import os

data_dir = r'd:\ModelRandomForest\data'
files = {
    'elevasi': 'elevasi_kecamatan_aceh_utara.csv',
    'ph': 'pH_tanah_kecamatan_aceh_utara.csv',
    'curah_hujan': 'curah_hujan_tahunan_kecamatan_aceh_utara_2025.csv',
    'suhu': 'suhu_tahunan_kecamatan_aceh_utara_2025.csv',
    'matahari': 'intensitas_matahari_tahunan_kecamatan_aceh_utara_2025.csv',
    'air': 'ketersediaan_air_kecamatan_aceh_utara_2025.csv'
}

# Base dataframe with the geographical info
df_base = None

for key, filename in files.items():
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    
    # Selecting relevant columns
    if key == 'elevasi':
        cols = ['ADM3_PCODE', 'kabupaten', 'kecamatan', 'provinsi', 'elevasi_mdpl']
    elif key == 'ph':
        cols = ['ADM3_PCODE', 'ph_tanah_mean']
    elif key == 'curah_hujan':
        cols = ['ADM3_PCODE', 'curah_hujan_tahunan']
    elif key == 'suhu':
        cols = ['ADM3_PCODE', 'suhu_tahunan_c']
    elif key == 'matahari':
        cols = ['ADM3_PCODE', 'intensitas_matahari_mj_m2_tahun']
    elif key == 'air':
        cols = ['ADM3_PCODE', 'ketersediaan_air_rootzone_persen']
    
    df_subset = df[cols]
    
    if df_base is None:
        df_base = df_subset
    else:
        df_base = pd.merge(df_base, df_subset, on='ADM3_PCODE', how='outer')

# Save the combined template
output_path = os.path.join(data_dir, 'data_gabungan_aceh_utara.csv')
df_base.to_csv(output_path, index=False)
print(f"Combined data saved to {output_path}")
