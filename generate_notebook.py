import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

# Section 1: Introduction
nb['cells'].append(new_markdown_cell("""
# Model Evaluation with Cross-Validation and Random Forest
## Project: Heart Disease Prediction

**Objectives:**
1. Apply K-fold and Stratified K-fold cross-validation.
2. Train and tune a Random Forest Classifer.
3. Compare Random Forest with SVM and Decision Trees.

**Dataset:**
- **Source**: `heart_disease_dataset.csv` (Local CSV file)
- **Size**: 400 samples, 17 Features
- **Features**: standard heart disease indicators + BMI, Smoker, Family History.
"""))

# Section 2: Imports
nb['cells'].append(new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
"""))

# Section 3: Data Loading
nb['cells'].append(new_markdown_cell("""
## 1. Data Preparation
Loading the dataset from `heart_disease_dataset.csv`.
"""))

nb['cells'].append(new_code_cell("""
# Load dataset
try:
    df = pd.read_csv('heart_disease_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'heart_disease_dataset.csv' not found. Please ensure the file is in the same directory.")
    # Fallback or exit logic could go here, but for this demo ensuring file presence is key
    
print(f"Dataset Shape: {df.shape}")
df.head()
"""))

# Section 4: EDA
nb['cells'].append(new_markdown_cell("""
## 2. Exploratory Data Analysis (EDA)
"""))

nb['cells'].append(new_code_cell("""
# Check info
df.info()
"""))

nb['cells'].append(new_code_cell("""
# Check distribution of target
sns.countplot(x='target', data=df)
plt.title('Target Distribution (0 = No Disease, 1 = Disease)')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()
"""))

nb['cells'].append(new_code_cell("""
# Preprocessing: Scale numerical features
scaler = StandardScaler()
X = df.drop('target', axis=1)
y = df['target']

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled.head()
"""))

# Section 5: Cross-Validation Implementation
nb['cells'].append(new_markdown_cell("""
## 3. Cross-Validation Implementation
We will demonstrate both K-Fold and Stratified K-Fold cross-validation.
"""))

nb['cells'].append(new_code_cell("""
# Initialize Models
rf = RandomForestClassifier(random_state=42)

# K-Fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf_scores = cross_val_score(rf, X_scaled, y, cv=kf, scoring='accuracy')

print(f"K-Fold CV Accuracy Scores: {kf_scores}")
print(f"Mean K-Fold Accuracy: {kf_scores.mean():.4f}")

# Stratified K-Fold CV (Better for classification)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
skf_scores = cross_val_score(rf, X_scaled, y, cv=skf, scoring='accuracy')

print(f"Stratified K-Fold CV Accuracy Scores: {skf_scores}")
print(f"Mean Stratified K-Fold Accuracy: {skf_scores.mean():.4f}")
"""))

# Section 6: Hyperparameter Tuning
nb['cells'].append(new_markdown_cell("""
## 4. Hyperparameter Tuning for Random Forest
Using GridSearchCV to find the optimal parameters.
"""))

nb['cells'].append(new_code_cell("""
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_scaled, y)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
"""))

# Section 7: Model Comparison
nb['cells'].append(new_markdown_cell("""
## 5. Model Comparison
Comparing Random Forest (Tuned) with SVM and Decision Tree.
"""))

nb['cells'].append(new_code_cell("""
# Define models
models = {
    "Random Forest (Tuned)": best_rf,
    "SVM": SVC(kernel='linear', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)

}

results = []

for name, model in models.items():
    res = {}
    res['Model'] = name
    
    # Perform Cross-Validation with multiple metrics
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(model, X_scaled, y, cv=10, scoring=scoring)
    
    res['Accuracy'] = cv_results['test_accuracy'].mean()
    res['Precision'] = cv_results['test_precision'].mean()
    res['Recall'] = cv_results['test_recall'].mean()
    res['F1 Score'] = cv_results['test_f1'].mean()
    
    results.append(res)

results_df = pd.DataFrame(results)
results_df
"""))

# Section 8: Visualization of Results
nb['cells'].append(new_markdown_cell("""
## 6. Performance Evaluation Report
"""))

nb['cells'].append(new_code_cell("""
# Melt dataframe for plotting
results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(x="Metric", y="Score", hue="Model", data=results_melted, palette="viridis")
plt.title("Model Performance Comparison (Cross-Validation)")
plt.ylim(0, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

print("Detailed Comparison Table:")
print(results_df)
"""))

# Final Save
with open('c:/Project/CrossValidation-RandomForest/Model_Evaluation.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully: Model_Evaluation.ipynb")
