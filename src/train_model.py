import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_rf_model(data_path):
    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Define features and target
    # We want to identify the environment (Kecamatan) from the attributes
    features = [
        'pH_Tanah', 
        'Suhu_C', 
        'Curah_Hujan_mm', 
        'Elevasi_mdpl', 
        'Ketersediaan_Air', 
        'Intensitas_Matahari_jam'
    ]
    X = df[features]
    y = df['Kecamatan_Encoded']
    
    # 1. Split data into 80/20 train-test ratio
    # Stratified split to ensure each kecamatan is in both sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Train RandomForestClassifier
    print("Training RandomForestClassifier to identify Kecamatan...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 3. Evaluation
    print("Evaluating model...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy (Kecamatan Identification): {accuracy * 100:.2f}%")
    
    # 4. Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    print("Model saved to 'models/random_forest_model.joblib'.")
    
    return rf_model, accuracy

if __name__ == "__main__":
    data_path = 'data/processed_dataset.csv'
    model, acc = train_rf_model(data_path)
    
    if acc >= 0.85:
        print("\nSuccess: Model accuracy is above 85%!")
    else:
        print("\nWarning: Model accuracy is below 85%.")
