# API Documentation: Horticulture Recommendation System

API ini menyediakan layanan prediksi varietas tanaman hortikultura terbaik berdasarkan input data lingkungan menggunakan algoritma **Random Forest**.

## 1. Menjalankan Server
Pastikan semua dependensi sudah terpasang, lalu jalankan perintah berikut di terminal:

```powershell
python src/api.py
```

Server akan berjalan secara default di: `http://localhost:8000`

---

## 2. Endpoint: Predict Varietas
Digunakan untuk mendapatkan daftar rekomendasi varietas tanaman berdasarkan data lingkungan.

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`

### Request Body
Format JSON yang harus dikirimkan:

```json
{
  "ph_tanah": 6.46,
  "suhu_c": 24.2,
  "curah_hujan_mm": 1763.0,
  "elevasi_mdpl": 255.0,
  "ketersediaan_air": "Rendah",
  "intensitas_matahari_jam": 6.4
}
```

| Field | Tipe | Deskripsi |
| :--- | :--- | :--- |
| `ph_tanah` | float | Nilai pH tanah (contoh: 6.5) |
| `suhu_c` | float | Suhu udara dalam Celcius (contoh: 28.5) |
| `curah_hujan_mm` | float | Curah hujan tahunan dalam mm |
| `elevasi_mdpl` | float | Ketinggian lokasi (mdpl) |
| `ketersediaan_air` | string | "Rendah", "Sedang", atau "Tinggi" |
| `intensitas_matahari_jam` | float | Lama penyinaran matahari per hari |

### Response Example (Success)
```json
{
  "status": "success",
  "identified_location": "Baktiya",
  "recommendations": [
    "Bayam Varietas Maestro",
    "Cabe Besar Varietas Tanjung 2",
    "Cabe Keriting Varietas TM 999",
    "Cabe Rawit Varietas Dewata F1",
    "Kacang Panjang Varietas Parade Tavi",
    "Kangkung Varietas Bangkok LP-1",
    "Ketimun Varietas Hercules F1",
    "Semangka Varietas Crimson Sweet",
    "Terung Varietas Mustang F1",
    "Tomat Varietas Servo F1"
  ],
  "raw_data": {
    "Bayam": "Maestro",
    "Cabe Besar": "Tanjung 2",
    "Cabe Keriting": "TM 999",
    "Cabe Rawit": "Dewata F1",
    "Kacang Panjang": "Parade Tavi",
    "Kangkung": "Bangkok LP-1",
    "Ketimun": "Hercules F1",
    "Semangka": "Crimson Sweet",
    "Terung": "Mustang F1",
    "Tomat": "Servo F1"
  }
}
```

---

## 3. Dokumentasi Interaktif (Swagger)
FastAPI secara otomatis menyediakan dokumentasi yang bisa langsung dicoba melalui browser:

1. Jalankan API.
2. Buka: [http://localhost:8000/docs](http://localhost:8000/docs)
3. Klik tombol **"Try it out"** pada endpoint `/predict`.

---

## 4. Error Codes
- `400 Bad Request`: Input `ketersediaan_air` tidak sesuai (harus Rendah/Sedang/Tinggi).
- `422 Unprocessable Entity`: Tipe data input salah (misal: mengirim string di kolom pH).
- `500 Internal Server Error`: Terjadi kesalahan pada server atau model belum dilatih.
