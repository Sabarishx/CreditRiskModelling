# CreditRiskModelling - Probability of Default (PD Model)

# Credit Risk Modelling – Probability of Default (PD) Prediction

## Overview

This project focuses on developing and evaluating machine learning models for **Credit Risk Assessment** by predicting the **Probability of Default (PD)** of loan applicants.

The repository demonstrates an end-to-end credit risk workflow including:

* Exploratory Data Analysis (EDA)
* Data Cleaning & Feature Engineering
* Missing Value Treatment
* Outlier Handling
* Categorical Encoding
* Feature Scaling
* Class Imbalance Treatment using SMOTE
* PD Model Development
* Model Evaluation
* Model Serialization
* Real-Time Prediction Testing
* Model Monitoring Dashboard

The objective is to identify high-risk borrowers and support better lending decisions through data-driven risk assessment.

---

## Business Problem

Financial institutions face significant losses due to loan defaults. Traditional rule-based approaches may not capture complex borrower behavior.

This project builds machine learning models capable of estimating the likelihood that a borrower will default on a loan, enabling:

* Better credit approval decisions
* Reduced default rates
* Improved portfolio quality
* Risk-based pricing strategies
* Regulatory and risk management support

---

## Repository Structure

```text
CreditRiskModelling/
│
├── EDA.ipynb
│   ├── Data Exploration
│   ├── Missing Value Analysis
│   ├── Outlier Detection
│   └── Feature Understanding
│
├── PreProcessing_PD_Model.ipynb
│   ├── Data Cleaning
│   ├── Feature Engineering
│   ├── Scaling
│   ├── SMOTE
│   └── Model Preparation
│
├── PreProcessing_PD_Model.py
│   └── Reusable preprocessing pipeline
│
├── Realtime_Dataset_ModelTesting.ipynb
│   └── Real-time scoring and model testing
│
├── Automated Pipeline.ipynb
│   └── Automated prediction workflow
│
├── Logistic_PD_Model.pkl
│   └── Trained Logistic Regression Model
│
├── XGBoost_PD_Model.pkl
│   └── Trained XGBoost Model
│
├── Model Monitoring.pbix
│   └── Power BI Monitoring Dashboard
│
├── pd_predictions.xlsx
│   └── Prediction Output File
│
└── README.md
```

---

## Dataset Features

The model uses borrower-related information such as:

### Applicant Information

* Age
* Income
* Employment Length
* Home Ownership Status

### Loan Information

* Loan Amount
* Loan Intent
* Interest Rate
* Loan Grade

### Credit History

* Historical Defaults
* Credit History Length

### Target Variable

```text
loan_status

0 = Non-Default
1 = Default
```

---

## Data Preprocessing Pipeline

### Missing Value Treatment

* Median imputation for interest rate variables.

### Outlier Handling

Records are filtered based on business rules:

```python
person_age <= 90
person_emp_length <= 45
```

### Feature Engineering

Categorical variables transformed using:

```python
pd.get_dummies()
```

Examples:

* Home Ownership
* Loan Intent

### Encoding

Binary conversion:

```text
Y → 1
N → 0
```

### Feature Scaling

```python
StandardScaler()
```

Used to normalize numerical variables before modeling.

### Class Imbalance Handling

Implemented using:

```python
SMOTE
```

to generate synthetic minority-class samples and improve model performance.

---

## Machine Learning Models

### Logistic Regression

Advantages:

* Highly interpretable
* Industry-standard for credit risk
* Regulatory-friendly

### Random Forest

Advantages:

* Captures non-linear relationships
* Robust against overfitting

### XGBoost

Advantages:

* High predictive power
* Handles complex interactions
* Industry-leading performance

---

## Model Evaluation Metrics

The following metrics are used for performance assessment:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Precision-Recall Curve

Example:

```python
accuracy_score()
precision_score()
recall_score()
f1_score()
confusion_matrix()
```

---

## Real-Time Prediction Workflow

1. Load new borrower data
2. Apply preprocessing pipeline
3. Load trained model
4. Generate default predictions
5. Export prediction results

Models are loaded using:

```python
pickle.load()
```

Example:

```python
with open('Logistic_PD_Model.pkl','rb') as file:
    model = pickle.load(file)
```

---

## Model Monitoring

The repository includes a Power BI dashboard:

```text
Model Monitoring.pbix
```

Possible monitoring areas include:

* Prediction Distribution
* Default Rate Trends
* Portfolio Risk Segmentation
* Model Performance Tracking
* Data Quality Monitoring

---

## Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* Power BI

### Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)

### Model Deployment

* Pickle

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sabarishx/CreditRiskModelling.git
```

Navigate to project folder:

```bash
cd CreditRiskModelling
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

---

## Usage

Run preprocessing:

```python
from PreProcessing_PD_Model import ModelPreProcessing
```

Load model:

```python
import pickle

with open("Logistic_PD_Model.pkl", "rb") as file:
    model = pickle.load(file)
```

Generate predictions:

```python
predictions = model.predict(features)
```

---

## Future Enhancements

* Probability Scorecard Development
* WOE & IV Feature Engineering
* PD Calibration
* Basel-Compliant Scorecards
* Explainable AI (SHAP)
* Model Drift Detection
* Automated Retraining Pipeline
* IFRS9/ECL Integration

## Author

**Sabarish Venkatesan**
LinkedIn: https://www.linkedin.com/in/sabarish-venkatesan-8b2779166/

## License

This project is licensed under the MIT License.

<img width="1877" height="1008" alt="image" src="https://github.com/user-attachments/assets/9625570a-97d8-4a6d-bac2-a35ead2f2dc7" />
