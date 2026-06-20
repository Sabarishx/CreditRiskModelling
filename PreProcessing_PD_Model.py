#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[72]:


def ModelPreProcessing(df):
    # 1. Fix SettingWithCopyWarning by explicitly creating a deep copy after filtering
    df_proc = df[(df['person_age'] <= 90) & (df['person_emp_length'] <= 45)].copy()
    
    # 2. Fill missing values safely
    df_proc.fillna({'loan_int_rate': df_proc['loan_int_rate'].median()}, inplace=True)
    
    # 3. Fix the ValueError: Drop loan_grade and re-assign it to the dataframe
    df_proc = df_proc.drop(columns=['loan_grade'], errors='ignore')

    # 4. Generate dummy variables and ensure we drop original columns to avoid text errors
    # Using drop_first=True makes these numeric binary flags
    person_home_ownership = pd.get_dummies(df_proc['person_home_ownership'], drop_first=True).astype(int)
    loan_intent = pd.get_dummies(df_proc['loan_intent'], drop_first=True).astype(int)
    
    # Isolate targets and numeric features explicitly
    target = df_proc['loan_status'].copy()
    
    # Drop all categorical and target columns to isolate purely numeric fields for scaling
    data_to_scale = df_proc.drop(columns=['person_home_ownership', 'loan_intent', 'cb_person_default_on_file', 'loan_status'], errors='ignore')

    # 5. Scale purely numeric features
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data_to_scale)
    
    # Rebuild the scaled dataframe using the clean index to prevent NaN misalignment
    scaled_numeric_df = pd.DataFrame(scaled_array, columns=data_to_scale.columns, index=df_proc.index)
    
    # 6. Recombine all features using the unified index
    features = pd.concat([scaled_numeric_df, person_home_ownership, loan_intent], axis=1)
    features['cb_person_default_on_file'] = np.where(df_proc['cb_person_default_on_file'] == 'Y', 1, 0)
    
    # 7. Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    balanced_features, balanced_target = smote.fit_resample(features, target)
    
    return target, features, balanced_features, balanced_target, data_to_scale


# In[ ]:




