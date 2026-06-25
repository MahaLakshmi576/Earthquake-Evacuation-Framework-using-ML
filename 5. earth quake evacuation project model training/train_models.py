import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
import json

def train_models():
    # Load data
    df = pd.read_csv('earthquake_data.csv')
    
    # Preprocessing
    le = LabelEncoder()
    df['SoilType'] = le.fit_transform(df['SoilType'])
    
    # Save label encoder
    with open('soil_le.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    X = df[['Magnitude', 'Depth', 'Distance', 'BuildingAge', 'PopulationDensity', 'SoilType']]
    y = df['RiskLabel']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Random Forest (Target ~90%)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    
    # 2. Linear Model (Logistic Regression) (Target ~75%)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    
    # Save models
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open('linear_model.pkl', 'wb') as f:
        pickle.dump(lr, f)
        
    # Save metrics for dashboard
    metrics = {
        'rf_accuracy': round(rf_acc * 100, 2),
        'rf_f1': round(rf_f1, 4),
        'lr_accuracy': round(lr_acc * 100, 2),
        'lr_f1': round(lr_f1, 4),
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Linear Model Accuracy: {lr_acc:.4f}")
    print("Models and metrics saved.")

if __name__ == "__main__":
    train_models()
