import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('data/processed_dataset.csv')
features = [
    'pH_Tanah', 'Suhu_C', 'Curah_Hujan_mm', 'Elevasi_mdpl', 
    'Ketersediaan_Air', 'Intensitas_Matahari_jam', 'Nama_Tanaman_Encoded'
]
X = df[features]
y = df['Nama_Varietas_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

print("\nSample Predictions vs True Values:")
results = pd.DataFrame({'True': y_test, 'Pred': y_pred})
print(results.head(10))

# Check if Pred values are even in the training set
print("\nUnique Predicted Values:", sorted(list(set(y_pred))))
print("Unique True Values:", sorted(list(set(y_test))))
