import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

def evaluate_saved_model():
    # 1. Path Configuration
    data_path = 'data/processed_dataset.csv'
    model_path = 'models/random_forest_model.joblib'
    le_kecamatan_path = 'models/le_kecamatan.joblib'

    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(le_kecamatan_path):
        print("Error: Model atau Encoder tidak ditemukan. Jalankan 'python src/train_model.py' terlebih dahulu.")
        return

    # 2. Load Data and Model
    print("--- Evaluasi Model Random Forest ---")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    le_kecamatan = joblib.load(le_kecamatan_path)

    # 3. Define features and target (Must match train_model.py)
    features = [
        'pH_Tanah', 
        'Suhu_C', 
        'Curah_Hujan_mm', 
        'Elevasi_mdpl', 
        'Ketersediaan_Air', 
        'Intensitas_Matahari_jam'
    ]
    X = df[features]
    y_true = df['Kecamatan_Encoded']

    # 4. Make Predictions
    print("Making predictions on the entire dataset...")
    y_pred = model.predict(X)

    # 5. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get class names for the report
    target_names = le_kecamatan.classes_

    print("\n" + "="*50)
    print(f"HASIL EVALUASI MODEL")
    print("="*50)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Optional: Show a few sample comparisons
    print("\nSample Perbandingan (5 data pertama):")
    comparison_df = pd.DataFrame({
        'Actual Kecamatan': le_kecamatan.inverse_transform(y_true[:5]),
        'Predicted Kecamatan': le_kecamatan.inverse_transform(y_pred[:5])
    })
    print(comparison_df)

if __name__ == "__main__":
    evaluate_saved_model()
