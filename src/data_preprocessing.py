import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def _load_raw(file_path):
    if file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = _load_raw(file_path)

    # 1. Handle Missing Values in Suhu_C
    missing_suhu_count = df['Suhu_C'].isnull().sum()
    if missing_suhu_count > 0:
        suhu_median = df['Suhu_C'].median()
        print(f"Handling missing values: Imputing {missing_suhu_count} missing values in 'Suhu_C' with median {suhu_median:.2f}°C...")
        df['Suhu_C'] = df['Suhu_C'].fillna(suhu_median)
    else:
        print("No missing values found in 'Suhu_C'.")

    # 2. Initialize LabelEncoder for target Nama_Varietas
    print("Encoding target 'Nama_Varietas'...")
    le_varietas = LabelEncoder()
    df['Nama_Varietas_Encoded'] = le_varietas.fit_transform(df['Nama_Varietas'])

    # 3. Initialize LabelEncoder for Nama_Tanaman
    print("Encoding feature 'Nama_Tanaman'...")
    le_tanaman = LabelEncoder()
    df['Nama_Tanaman_Encoded'] = le_tanaman.fit_transform(df['Nama_Tanaman'])

    # 4. Initialize LabelEncoder for Kecamatan
    print("Encoding feature 'Kecamatan'...")
    le_kecamatan = LabelEncoder()
    df['Kecamatan_Encoded'] = le_kecamatan.fit_transform(df['Kecamatan'])

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save encoders for future use in prediction/API
    joblib.dump(le_varietas, 'models/le_varietas.joblib')
    joblib.dump(le_tanaman, 'models/le_tanaman.joblib')
    joblib.dump(le_kecamatan, 'models/le_kecamatan.joblib')
    print("Encoders saved successfully to 'models/' directory.")

    return df, le_varietas, le_tanaman, le_kecamatan


def preprocess_train_test(train_path, test_path):
    """Fit encoders on the combined train+test data (so both splits share the
    same label space) then transform each split separately, preserving the
    original train/test assignment instead of re-splitting."""
    print(f"Loading train data from {train_path}...")
    train_df = _load_raw(train_path)
    print(f"Loading test data from {test_path}...")
    test_df = _load_raw(test_path)

    for name, df in (("train", train_df), ("test", test_df)):
        missing_suhu_count = df['Suhu_C'].isnull().sum()
        if missing_suhu_count > 0:
            suhu_median = df['Suhu_C'].median()
            print(f"[{name}] Imputing {missing_suhu_count} missing 'Suhu_C' values with median {suhu_median:.2f}°C...")
            df['Suhu_C'] = df['Suhu_C'].fillna(suhu_median)

    combined = pd.concat([train_df, test_df], ignore_index=True)

    print("Encoding target 'Nama_Varietas'...")
    le_varietas = LabelEncoder()
    le_varietas.fit(combined['Nama_Varietas'])

    print("Encoding feature 'Nama_Tanaman'...")
    le_tanaman = LabelEncoder()
    le_tanaman.fit(combined['Nama_Tanaman'])

    print("Encoding feature 'Kecamatan'...")
    le_kecamatan = LabelEncoder()
    le_kecamatan.fit(combined['Kecamatan'])

    for df in (train_df, test_df):
        df['Nama_Varietas_Encoded'] = le_varietas.transform(df['Nama_Varietas'])
        df['Nama_Tanaman_Encoded'] = le_tanaman.transform(df['Nama_Tanaman'])
        df['Kecamatan_Encoded'] = le_kecamatan.transform(df['Kecamatan'])

    os.makedirs('models', exist_ok=True)
    joblib.dump(le_varietas, 'models/le_varietas.joblib')
    joblib.dump(le_tanaman, 'models/le_tanaman.joblib')
    joblib.dump(le_kecamatan, 'models/le_kecamatan.joblib')
    print("Encoders saved successfully to 'models/' directory.")

    combined_processed = pd.concat([train_df, test_df], ignore_index=True)

    return train_df, test_df, combined_processed, le_varietas, le_tanaman, le_kecamatan


if __name__ == "__main__":
    train_path = 'data3/train_noise.xlsx'
    test_path = 'data3/test_noise.xlsx'

    train_df, test_df, combined_df, le_v, le_t, le_k = preprocess_train_test(train_path, test_path)

    os.makedirs('data3', exist_ok=True)
    train_df.to_csv('data3/processed_train.csv', index=False)
    test_df.to_csv('data3/processed_test.csv', index=False)
    combined_df.to_csv('data3/processed_dataset.csv', index=False)
    print("Processed datasets saved to 'data3/processed_train.csv', 'data3/processed_test.csv' and 'data3/processed_dataset.csv'.")

    print("\nPre-processing Summary:")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Unique Plants encoded: {len(le_t.classes_)} ({', '.join(le_t.classes_)})")
    print(f"Unique Districts (Kecamatan) encoded: {len(le_k.classes_)}")
    print(f"Unique Varieties (Target Class) encoded: {len(le_v.classes_)}")
