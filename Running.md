# Panduan Menjalankan Sistem (Workflow)

Ikuti langkah-langkah di bawah ini secara berurutan jika Anda ingin mengulangi proses dari awal atau jika ada pembaruan data pada `data/dataset_training_random_forest.csv`.

## Langkah 1: Preprocessing Data
Membersihkan data mentah dan menyiapkan encoder untuk kategori.
```powershell
python src/data_preprocessing.py
```
**Output:**
- `data/processed_dataset.csv`
- `models/le_varietas.joblib`
- `models/le_tanaman.joblib`
- `models/le_kecamatan.joblib`

## Langkah 2: Pelatihan Model (Training)
Melatih model Random Forest menggunakan data hasil preprocess.
```powershell
python src/train_model.py
```
**Output:**
- `models/random_forest_model.joblib`
- Menampilkan akurasi model di terminal.

## Langkah 3: Antarmuka Prediksi (CLI)
(Opsional) Digunakan untuk menguji input data lingkungan secara manual di terminal.
```powershell
python src/predict.py
```

## Langkah 4: Menjalankan REST API
Langkah terakhir untuk mengaktifkan server agar sistem bisa diakses melalui URL.
```powershell
python src/api.py
```
**Akses API:**
- Base URL: `http://localhost:8000`
- Dokumentasi Swagger: `http://localhost:8000/docs`

---
*Catatan: Selalu jalankan perintah di atas dari root folder proyek `ModelRandomForest`.*
