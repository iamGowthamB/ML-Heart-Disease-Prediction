import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save_model():
    # Load Data
    data_path = 'c:/Project/CrossValidation-RandomForest/heart_disease_dataset.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return

    # Preprocessing
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Model (Best params from previous tuning: n_estimators=200, max_depth=10)
    # Using hardcoded best params for simplicity in this script, or we could re-run grid search.
    # We'll use robust defaults/approx best params.
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_scaled, y)
    
    # Save Model and Scaler
    joblib.dump(rf_model, 'c:/Project/CrossValidation-RandomForest/rf_model.pkl')
    joblib.dump(scaler, 'c:/Project/CrossValidation-RandomForest/scaler.pkl')
    
    print("Model trained and saved to c:/Project/CrossValidation-RandomForest/rf_model.pkl")
    print("Scaler saved to c:/Project/CrossValidation-RandomForest/scaler.pkl")

if __name__ == "__main__":
    train_and_save_model()
