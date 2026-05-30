import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_rf_model(data_path):
    print(f"Loading preprocessed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Define features (environmental factors) and target (Kecamatan_Encoded)
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
    
    # 2. Split data into 80/20 train-test ratio with stratification
    print("Splitting data into stratified train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # 3. Train RandomForestClassifier
    print("Training RandomForestClassifier to predict Kecamatan...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 4. Evaluate on test set
    print("Evaluating model accuracy on test set...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy (Kecamatan Classification): {accuracy * 100:.2f}%")
    
    # 5. Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    print("Model saved to 'models/random_forest_model.joblib'.")
    
    return rf_model, accuracy

if __name__ == "__main__":
    data_path = 'data/processed_dataset.csv'
    model, acc = train_rf_model(data_path)
    
    if acc >= 0.90:
        print("\nSuccess: Model accuracy is excellent (above 90%)!")
    elif acc >= 0.80:
        print("\nSuccess: Model accuracy is good (above 80%).")
    else:
        print("\nWarning: Model accuracy is below 80%. Consider tuning hyperparameters.")
