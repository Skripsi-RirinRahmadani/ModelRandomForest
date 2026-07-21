import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

FEATURES = [
    'pH_Tanah',
    'Suhu_C',
    'Curah_Hujan_mm',
    'Elevasi_mdpl'
]


def train_rf_model(train_path, test_path):
    print(f"Loading preprocessed train data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading preprocessed test data from {test_path}...")
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURES]
    y_train = train_df['Kecamatan_Encoded']
    X_test = test_df[FEATURES]
    y_test = test_df['Kecamatan_Encoded']

    # Train RandomForestClassifier
    print("Training RandomForestClassifier to predict Kecamatan...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Evaluate on the provided test set
    print("Evaluating model accuracy on test set...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy (Kecamatan Classification): {accuracy * 100:.2f}%")

    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    print("Model saved to 'models/random_forest_model.joblib'.")

    return rf_model, accuracy


if __name__ == "__main__":
    train_path = 'data3/processed_train.csv'
    test_path = 'data3/processed_test.csv'
    model, acc = train_rf_model(train_path, test_path)

    if acc >= 0.90:
        print("\nSuccess: Model accuracy is excellent (above 90%)!")
    elif acc >= 0.80:
        print("\nSuccess: Model accuracy is good (above 80%).")
    else:
        print("\nWarning: Model accuracy is below 80%. Consider tuning hyperparameters.")
