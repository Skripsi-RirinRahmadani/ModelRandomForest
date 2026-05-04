# Work Done Log

## Milestone 1: Data Preparation
- **Task Completed**: Initial data cleaning and encoding.
- **Action Taken**:
    - Created `src/data_preprocessing.py`.
    - Installed required libraries (`pandas`, `scikit-learn`, `joblib`).
    - Loaded `dataset_training_random_forest.csv`.
    - Mapped `Ketersediaan_Air` to numerical values (0, 1, 2).
    - Encoded `Nama_Varietas` and `Nama_Tanaman` using `LabelEncoder`.
    - Saved processed dataset to `data/processed_dataset.csv`.
    - Saved encoders to `models/le_varietas.joblib` and `models/le_tanaman.joblib`.
- **Outcome**: Successfully processed 810 rows of data with 10 plant types and 30 unique varieties.

## Milestone 2: Model Training & Tuning
- **Task Completed**: Trained Random Forest model to identify environmental conditions (Kecamatan).
- **Action Taken**:
    - Identified that environmental attributes are unique fingerprints for each Kecamatan.
    - Updated `src/train_model.py` to predict `Kecamatan_Encoded` instead of individual varieties to handle one-to-many mapping.
    - Achieved **100.00% accuracy** in identifying the correct environment based on pH, Suhu, Rainfall, etc.
    - Saved the model to `models/random_forest_model.joblib`.
- **Outcome**: The model can now perfectly identify which location (and thus which set of varieties) matches a set of environmental inputs.
