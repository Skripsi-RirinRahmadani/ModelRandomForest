# MVP Plan: Random Forest Plant Recommendation System

This document outlines the Minimum Viable Product (MVP) for the Random Forest system to recommend horticulture plant varieties.

## 1. Objectives
- Successfully train a Random Forest model using the provided Aceh Utara environmental dataset.
- Provide a simple interface (Python script/CLI) to input environmental parameters.
- Output a list of recommended varieties for different plant types (Ketimun, Cabai, Tomat, etc.).

## 2. Key Features
- **Data Preprocessor**: Automatically cleans and encodes features from `dataset_training_random_forest.csv`.
- **Core ML Model**: A trained `RandomForestClassifier` with optimized parameters.
- **Recommendation Engine**: A function that filters and predicts the best variety for each plant category based on a single set of inputs.
- **Formatted Output**: Results displayed in a clean, readable format as per the reference image.

## 3. Technical Stack
- **Language**: Python 3.x
- **Libraries**: 
    - `pandas`: Data manipulation.
    - `scikit-learn`: Model training and evaluation.
    - `joblib`: Model serialization (saving/loading).

## 4. Development Milestones

### Milestone 1: Data Preparation (Day 1)
- [x] Load `dataset_training_random_forest.csv`.
- [x] Map `Ketersediaan_Air` to numerical values.
- [x] Initialize `LabelEncoder` for target `Nama_Varietas`.

### Milestone 2: Model Training & Tuning (Day 1-2)
- [x] Split data into 80/20 train-test ratio.
- [x] Train `RandomForestClassifier`.
- [x] Achieve at least 85% accuracy on the test set.
- [x] Save the model and encoders using `joblib`.

### Milestone 3: Interface Development (Day 2)
- [x] Create `predict.py` to take user input (pH, Suhu, etc.).
- [x] Implement the "Top Recommendation" logic for each plant type.
- [x] Format the output string.

### Milestone 4: API Development (Day 3)
- [x] Build a REST API using **FastAPI**.
- [x] Create a `/predict` endpoint that receives JSON input.
- [x] Implement model loading logic within the API.
- [x] Document the API using Swagger/OpenAPI (automatically provided by FastAPI).

## 5. Success Criteria
- The system correctly identifies varieties for at least 5 different plant types from a single input.
- Input validation handles out-of-range environmental data gracefully.
