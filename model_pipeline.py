import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def generate_mock_credit_data(num_records=5000):
    """Generates synthetic loan applicant data for portfolio demonstration."""
    print("[*] Generating synthetic credit data...")
    np.random.seed(42)
    
    data = {
        'Income': np.random.normal(60000, 20000, num_records),
        'Credit_Score': np.random.normal(650, 80, num_records),
        'Debt_to_Income': np.random.uniform(0.1, 0.6, num_records),
        'Loan_Amount': np.random.normal(15000, 5000, num_records),
        'Employment_Years': np.random.poisson(5, num_records)
    }
    df = pd.DataFrame(data)
    
    # Simulate default logic: Low score + high debt = higher chance of default (1)
    risk_factor = (df['Debt_to_Income'] * 2) - (df['Credit_Score'] / 1000)
    df['Default_Risk'] = (risk_factor > np.median(risk_factor)).astype(int)
    
    return df

def train_risk_model():
    df = generate_mock_credit_data()
    
    X = df.drop('Default_Risk', axis=1)
    y = df['Default_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[*] Training LightGBM Risk Classifier...")
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print("\n--- Model Evaluation ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    print(classification_report(y_test, preds))
    
    # Save Model
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/lgbm_risk_engine.pkl')
    print("[*] Model successfully serialized to model/lgbm_risk_engine.pkl")

if __name__ == "__main__":
    train_risk_model()
