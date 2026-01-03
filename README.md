**Problem Statement**

Customer churn is one of the most critical challenges faced by telecom companies.
Acquiring a new customer is significantly more expensive than retaining an existing one.
However, telecom operators often do not know why a customer is leaving, which limits proactive retention strategies.

**Objectives**

Predict customer churn with high accuracy
Generate individual-level explanations for predictions
Produce actionable insights for business teams
Ensure transparency and trust using Explainable AI (XAI)

Dataset
Source: IBM Telco Customer Churn Dataset (Kaggle)
🔗 Dataset Link:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset Overview
7043 customers
21 features
Includes:
Customer demographics
Service subscriptions
Contract & billing information
Target variable: Churn (Yes/No)


Design
Data Ingestion
      ↓
Data Cleaning & Encoding
      ↓
Feature Engineering
      ↓
LightGBM Churn Prediction Model
      ↓
SHAP Explainability Engine
      ↓
Churn Scores + Why-Reason Analysis
      ↓
CSV Output / API 


Technologies Used
1️.Python
2️.LightGBM
3️.SHAP (SHapley Additive Explanations)
4️.Pandas & NumPy
5️.Matplotlib / Seaborn
6️.FastAPI
Deploys the model as a REST API
Allows real-time churn prediction via /score endpoint


Assumptions

1.Historical behavior predicts future churn
The model assumes that past customer behavior (usage, billing, contract type, services subscribed) is indicative of future churn behavior.

2.Dataset is representative of real-world telecom customers
The IBM Telco Customer Churn dataset is assumed to reflect realistic customer demographics, service usage patterns, and churn characteristics found in telecom industries.

3.No major external market disruptions
The model assumes no sudden external changes such as major price hikes, regulatory changes, mergers, or service outages that could drastically affect churn patterns.

4.Churn definition is binary and accurate
Churn is assumed to be correctly labeled as a binary outcome (Yes / No), and customers marked as churned have definitively discontinued service.

5.Missing values are minimal and manageable
Any missing or invalid values (e.g., blank TotalCharges) can be safely handled through preprocessing without distorting customer behavior.
