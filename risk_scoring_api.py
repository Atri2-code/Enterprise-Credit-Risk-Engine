import joblib
import pandas as pd

class CreditRiskEngine:
    def __init__(self, model_path='model/lgbm_risk_engine.pkl'):
        try:
            self.model = joblib.load(model_path)
            print("[*] Risk Engine Loaded Successfully.")
        except FileNotFoundError:
            print("[!] Model not found. Please run model_pipeline.py first.")

    def score_applicant(self, applicant_data):
        """Scores a single applicant and returns probability of default."""
        df = pd.DataFrame([applicant_data])
        probability = self.model.predict_proba(df)[0][1]
        
        # Business Logic thresholds
        if probability > 0.70:
            decision = "REJECT - High Risk"
        elif probability > 0.40:
            decision = "MANUAL REVIEW - Medium Risk"
        else:
            decision = "APPROVE - Low Risk"
            
        return {
            "Default_Probability": round(probability, 4),
            "Business_Decision": decision
        }

if __name__ == "__main__":
    engine = CreditRiskEngine()
    
    # Test with a sample applicant
    sample_applicant = {
        'Income': 45000,
        'Credit_Score': 580, # Low score
        'Debt_to_Income': 0.45, # High debt
        'Loan_Amount': 20000,
        'Employment_Years': 2
    }
    
    print("\n--- Scoring New Applicant ---")
    print(f"Applicant Profile: {sample_applicant}")
    result = engine.score_applicant(sample_applicant)
    print(f"Engine Decision: {result['Business_Decision']} (Risk Score: {result['Default_Probability']})")
