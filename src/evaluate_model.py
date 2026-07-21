import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import numpy as np

def evaluate_saved_model():
    # 1. Path Configuration
    test_data_path = 'data3/processed_test.csv'
    model_path = 'models/random_forest_model.joblib'
    le_kecamatan_path = 'models/le_kecamatan.joblib'

    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(le_kecamatan_path):
        print("Error: Model atau Encoder tidak ditemukan. Jalankan 'python src/train_model.py' terlebih dahulu.")
        return

    # 2. Load Data, Model, and Encoders
    print("--- Evaluasi Model Random Forest (Klasifikasi Kecamatan) ---")
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path)

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    le_kecamatan = joblib.load(le_kecamatan_path)

    # 3. Define features and target (Must match train_model.py)
    features = [
        'pH_Tanah',
        'Suhu_C',
        'Curah_Hujan_mm',
        'Elevasi_mdpl'
    ]
    X_test = df[features]
    y_test = df['Kecamatan_Encoded']

    # 4. Make Predictions on Test Set
    print("Making predictions on the Test set...")
    y_pred = model.predict(X_test)

    # 6. Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Test set may not contain every Kecamatan class seen during training,
    # so restrict target_names to the labels actually present.
    present_labels = sorted(set(y_test.unique()) | set(np.unique(y_pred)))
    target_names = le_kecamatan.inverse_transform(present_labels)

    print("\n" + "="*60)
    print(f"HASIL EVALUASI MODEL (TEST SET)")
    print("="*60)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=target_names, zero_division=0))
    
    # 7. Extract Feature Importances
    print("\n" + "="*60)
    print("ANALISIS FEATURE IMPORTANCE")
    print("="*60)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # Sort descending
    
    feature_importance_list = []
    print(f"{'Rank':<5} | {'Feature Name':<25} | {'Relative Importance':<20} | {'Percentage':<10}")
    print("-" * 70)
    for rank, i in enumerate(indices):
        name = features[i]
        val = importances[i]
        pct = val * 100
        print(f"{rank+1:<5} | {name:<25} | {val:<19.6f} | {pct:.2f}%")
        feature_importance_list.append((name, val))
        
    print("-" * 70)
    
    # Optional: Show a few sample comparisons
    print("\nSample Perbandingan Prediksi Lokasi (10 data pertama di Test Set):")
    comparison_df = pd.DataFrame({
        'Actual Kecamatan': le_kecamatan.inverse_transform(y_test[:10]),
        'Predicted Kecamatan': le_kecamatan.inverse_transform(y_pred[:10])
    })
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_saved_model()
