# 🏦 Enterprise Credit Risk Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-ff69b4?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

A predictive machine learning pipeline designed for the Financial Services and Retail Banking sector. This engine replaces manual underwriting processes by evaluating tabular financial data to score loan applicant risk, successfully quantifying the probability of default.

## 💼 Business Impact
* **Risk Mitigation:** Designed to reduce bad-debt exposure by programmatically identifying high-risk applicants before loan approval.
* **Operational Efficiency:** Translates raw machine learning probabilities into automated, actionable business routing (e.g., Auto-Approve, Manual Review, Auto-Reject), saving underwriting hours.

## ⚙️ Architecture & Pipeline
1. **Data Ingestion & Engineering:** Processes key financial indicators (Debt-to-Income, Credit Score, Loan Amount).
2. **Classification Engine:** Utilizes **LightGBM** (Light Gradient Boosting Machine) for high-speed, accurate tabular data classification.
3. **Scoring API:** A lightweight evaluation script that ingests a single applicant's JSON/Dict profile and outputs a localized business decision.

## 🚀 Quick Start
```bash
git clone [https://github.com/Atri2-code/enterprise-credit-risk-engine.git](https://github.com/Atri2-code/enterprise-credit-risk-engine.git)
cd enterprise-credit-risk-engine
pip install -r requirements.txt

# 1. Generate data and train the LightGBM model
python model_pipeline.py

# 2. Test the engine on a sample applicant
python risk_scoring_api.py
