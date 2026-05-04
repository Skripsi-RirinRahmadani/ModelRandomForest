import requests
import json

url = "http://localhost:8000/predict"
data = {
    "ph_tanah": 6.5,
    "suhu_c": 28.5,
    "curah_hujan_mm": 2100,
    "elevasi_mdpl": 150,
    "ketersediaan_air": "Tinggi",
    "intensitas_matahari_jam": 8.0
}

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
