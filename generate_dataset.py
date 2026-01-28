import pandas as pd
import numpy as np

def generate_heart_data_csv(n_samples=400):
    np.random.seed(42)  # Ensure reproducibility
    
    data = {
        'age': np.random.normal(54, 9, n_samples).astype(int),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.normal(131, 17, n_samples).astype(int),
        'chol': np.random.normal(246, 51, n_samples).astype(int),
        'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.normal(149, 23, n_samples).astype(int),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.round(np.random.gamma(1, 1, n_samples), 1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(1, 4, n_samples),
        # New Columns
        'bmi': np.round(np.random.normal(26, 4, n_samples), 1), # Body Mass Index
        'smoker': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]), # 0=No, 1=Yes
        'famhist': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]) # Family History
    }
    
    df = pd.DataFrame(data)
    
    # Generate target with influence from new columns too
    # Logic: Higher risk factors -> higher  score
    score = (df['age'] * 0.3) + \
            (df['cp'] * 5) + \
            (df['chol'] * 0.05) + \
            (df['exang'] * 5) + \
            (df['bmi'] * 0.5) + \
            (df['smoker'] * 3) + \
            (df['famhist'] * 2) - \
            (df['thalach'] * 0.1) + \
            np.random.normal(0, 10, n_samples)
            
    threshold = np.median(score)
    df['target'] = (score > threshold).astype(int)
    
    output_path = 'c:/Project/CrossValidation-RandomForest/heart_disease_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at: {output_path}")
    print(f"Shape: {df.shape}")
    return df

if __name__ == "__main__":
    generate_heart_data_csv()
