import json

notebook_path = "Random_Forest_Plant_Recommendation_Revamped.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Markdown Cell 0 (Header description)
source_0 = nb["cells"][0]["source"]
for idx, line in enumerate(source_0):
    if "(pH Tanah, Suhu, Curah Hujan, Elevasi, Air, Matahari)" in line:
        source_0[idx] = line.replace("(pH Tanah, Suhu, Curah Hujan, Elevasi, Air, Matahari)", "(pH Tanah, Suhu, Curah Hujan, Elevasi)")

# Update Code Cell 4 (Dataset path)
nb["cells"][4]["source"] = [
    "# Load dataset baru hasil generasi di folder data2/\n",
    "data_path = 'data2/dataset_training_random_forest_generated.csv'\n",
    "df_raw = pd.read_csv(data_path)\n",
    "print(f\"Total Baris Data: {len(df_raw)} baris\")\n",
    "df_raw.head()"
]
nb["cells"][4]["outputs"] = []

# Update Code Cell 8 (Preprocessing steps)
nb["cells"][8]["source"] = [
    "# A. Label Encoding untuk Target Kecamatan\n",
    "print(\"Melakukan Label Encoding pada Target 'Kecamatan'...\")\n",
    "le_kecamatan = LabelEncoder()\n",
    "df['Kecamatan_Encoded'] = le_kecamatan.fit_transform(df['Kecamatan'])\n",
    "\n",
    "# B. Label Encoding Tambahan untuk Nama_Tanaman dan Nama_Varietas\n",
    "le_tanaman = LabelEncoder()\n",
    "df['Nama_Tanaman_Encoded'] = le_tanaman.fit_transform(df['Nama_Tanaman'])\n",
    "\n",
    "le_varietas = LabelEncoder()\n",
    "df['Nama_Varietas_Encoded'] = le_varietas.fit_transform(df['Nama_Varietas'])\n",
    "\n",
    "# Simpan encoder ke folder models/\n",
    "os.makedirs('models', exist_ok=True)\n",
    "joblib.dump(le_kecamatan, 'models/le_kecamatan.joblib')\n",
    "joblib.dump(le_tanaman, 'models/le_tanaman.joblib')\n",
    "joblib.dump(le_varietas, 'models/le_varietas.joblib')\n",
    "print(\"Semua encoder berhasil disimpan ke folder 'models/'\")\n",
    "\n",
    "# Simpan dataset preprocessed\n",
    "os.makedirs('data2', exist_ok=True)\n",
    "df.to_csv('data2/processed_dataset.csv', index=False)\n",
    "print(\"Dataset preprocessed berhasil disimpan ke 'data2/processed_dataset.csv'\")"
]
nb["cells"][8]["outputs"] = []

# Update Code Cell 10 (Feature definition)
nb["cells"][10]["source"] = [
    "# Tentukan Fitur Lingkungan (X) dan Target Lokasi (y)\n",
    "feature_names = ['pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 'Elevasi_mdpl']\n",
    "X = df[feature_names]\n",
    "y = df['Kecamatan_Encoded']\n",
    "\n",
    "# Lakukan Stratified Split untuk menjaga keseimbangan persebaran data tiap Kecamatan\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, \n",
    "    test_size=0.2, \n",
    "    random_state=42, \n",
    "    stratify=y\n",
    ")\n",
    "\n",
    "print(f\"Jumlah Data Training : {len(X_train)} baris\")\n",
    "print(f\"Jumlah Data Testing  : {len(X_test)} baris\")"
]
nb["cells"][10]["outputs"] = []

# Update Code Cell 21 (Simulation and recommendation function)
nb["cells"][21]["source"] = [
    "def recommend_varieties(ph, suhu, curah_hujan, elevasi):\n",
    "    # 1. Siapkan input untuk prediksi\n",
    "    input_df = pd.DataFrame([[ph, suhu, curah_hujan, elevasi]], \n",
    "                            columns=feature_names)\n",
    "    \n",
    "    # 2. Prediksi probabilitas tiap kecamatan menggunakan model\n",
    "    probabilities = rf_model.predict_proba(input_df)[0]\n",
    "    kec_names = le_kecamatan.classes_\n",
    "    prob_map = dict(zip(kec_names, probabilities))\n",
    "    \n",
    "    best_kec = kec_names[np.argmax(probabilities)]\n",
    "    confidence = np.max(probabilities)\n",
    "    \n",
    "    # 3. Normalisasi Min-Max sebagai acuan jarak similarity\n",
    "    df_min = df[feature_names].min()\n",
    "    df_max = df[feature_names].max()\n",
    "    df_range = (df_max - df_min).replace(0, 1)\n",
    "    \n",
    "    normalized_input = (np.array([ph, suhu, curah_hujan, elevasi]) - df_min.values) / df_range.values\n",
    "    \n",
    "    # 4. Kelompokkan varietas untuk mendapatkan rata-rata parameter lingkungan pendukung\n",
    "    variety_env = df.groupby(['Nama_Tanaman', 'Nama_Varietas'])[feature_names].mean().reset_index()\n",
    "    variety_locs = df.groupby(['Nama_Tanaman', 'Nama_Varietas'])['Kecamatan'].unique().reset_index()\n",
    "    variety_data = variety_env.merge(variety_locs, on=['Nama_Tanaman', 'Nama_Varietas'])\n",
    "    \n",
    "    # 5. Hitung Blended Score\n",
    "    def calculate_score(row):\n",
    "        # A. Location Score (bobot lokasi)\n",
    "        location_score = sum(prob_map.get(k, 0) for k in row['Kecamatan'])\n",
    "        \n",
    "        # B. Similarity Score (jarak Euclidean)\n",
    "        variety_features = (row[feature_names].values - df_min.values) / df_range.values\n",
    "        dist = np.linalg.norm(normalized_input - variety_features)\n",
    "        similarity_score = 1 / (1 + dist)\n",
    "        \n",
    "        # C. Blended Score (60% Bobot Lokasi Model, 40% Kemiripan Iklim)\n",
    "        return (location_score * 0.6 + similarity_score * 0.4) * 0.98\n",
    "    \n",
    "    variety_data['Score'] = variety_data.apply(calculate_score, axis=1)\n",
    "    \n",
    "    # 6. Pilih varietas dengan nilai kecocokan tertinggi untuk masing-masing tanaman\n",
    "    recommendations = variety_data.sort_values('Score', ascending=False).groupby('Nama_Tanaman').head(1).reset_index()\n",
    "    \n",
    "    print(\"=\"*70)\n",
    "    print(\"HASIL SIMULASI SISTEM REKOMENDASI VARIETAS HORTIKULTURA\")\n",
    "    print(\"=\"*70)\n",
    "    print(f\"Kecamatan Terdekat Teridentifikasi : {best_kec}\")\n",
    "    print(f\"Tingkat Keyakinan Geografis Model : {confidence * 100:.2f}%\")\n",
    "    print(\"=\"*70)\n",
    "    print(\"\\nRekomendasi Komoditas Terbaik:\")\n",
    "    for _, row in recommendations.sort_values('Nama_Tanaman').iterrows():\n",
    "        print(f\"- {row['Nama_Tanaman']:<18}: Varietas {row['Nama_Varietas']:<18} (Kecocokan: {row['Score']*100:.2f}%)\")\n",
    "    print(\"=\"*70)\n",
    "\n",
    "# Simulasi input dari user\n",
    "recommend_varieties(\n",
    "    ph=6.46, \n",
    "    suhu=24.2, \n",
    "    curah_hujan=1763, \n",
    "    elevasi=255\n",
    ")"
]
nb["cells"][21]["outputs"] = []

# Clear outputs for other executable code cells to ensure a clean execution flow
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")
