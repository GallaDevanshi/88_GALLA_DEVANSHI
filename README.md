**Problem Statement**

Customer churn is one of the most critical challenges faced by telecom companies.
Acquiring a new customer is significantly more expensive than retaining an existing one.
However, telecom operators often do not know why a customer is leaving, which limits proactive retention strategies.


**Problem to Solve**

Predict which customers are likely to churn
Explain why each customer is predicted to churn
Enable data-driven retention decisions
This project builds an Explainable Machine Learning system that not only predicts churn but also provides clear, human-readable reasons for each prediction.


**Objectives**

Predict customer churn with high accuracy
Generate individual-level explanations for predictions
Produce actionable insights for business teams
Ensure transparency and trust using Explainable AI (XAI)

**Dataset**
Source: IBM Telco Customer Churn Dataset (Kaggle)
🔗 Dataset Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset Overview
7043 customers
21 features
Includes:
Customer demographics
Service subscriptions
Contract & billing information
Target variable: Churn (Yes/No)


**Design**

1.Data Ingestion:Load Telco churn dataset from Kaggle

2.Data Preprocessing

Handle missing values

Encode categorical variables

Convert target labels to binary format

3.Feature Engineering

Prepare numerical and categorical features

Normalize/transform where required

4.Model Training

Train LightGBM classifier

Optimize performance using validation data

5.Churn Prediction

Generate churn probabilities for all customers

Classify customers as high-risk or low-risk

6.Explainability Layer (SHAP)

Compute SHAP values

Generate waterfall plots for selected customers

Identify top churn-driving factors

7.Output Generation

Export churn scores to CSV

Save SHAP explanation plots

Provide business-friendly insights 


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
