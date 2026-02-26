# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:41:33 2026

@author: Amol Gaikwad
"""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV



# ==================*

# *LOAD DATASET*

# ==================

df = pd.read_csv("C:/Users/Amol Gaikwad/Data Science/employee_attritions.csv")

# =====================================================
# Target & Feature Separation
# =====================================================

df['Attrition'] = pd.to_numeric(df['Attrition'], errors='coerce')

customer_id = df['Employee ID']

# Drop customerID as it has no predictive value
df.drop('Employee ID', axis=1, inplace=True)


# =====================================================
#  FEATURE & TARGET SEPARATION
# =====================================================

X = df.drop('Attrition', axis=1)   # Features
y = df['Attrition']               # Target

# =====================================================
#  ENCODE CATEGORICAL VARIABLES
# =====================================================

le = LabelEncoder()

# Encode target variable (Yes=1, No=0)
y = le.fit_transform(y)

# Encode all categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

# =====================================================
# Train Test Split
# =====================================================
X_train, X_test, y_train, y_test, cust_train, cust_test = train_test_split(
    X, y, customer_id, test_size=0.3, random_state=42, stratify=y
)
# =====================================================
# XGBoost
# =====================================================

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)

# =====================================================
# Parameter grid
# =====================================================

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 5]
}

scoring = {
    'AUC': 'roc_auc',
    'Accuracy': 'accuracy',
    'F1': 'f1',
    
    
}

# =====================================================
# GridSearchCV
# =====================================================

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=scoring,
    refit='AUC',
    cv=4,
    n_jobs=-1,
    verbose=2
)

# =====================================================
# Fit model
# =====================================================

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# =====================================================
#      Best results
# =====================================================
print("Best Parameters:", grid_search.best_params_)
print("Best AUC:", grid_search.best_score_)
print("Classification Report:\n", classification_report(y_test,y_pred))
print("Best CV Score:\n",grid_search.best_score_)

# =====================================================
# View all GridSearch results
# =====================================================

cv_df = pd.DataFrame(grid_search.cv_results_)

cv_df = cv_df[['params', 'mean_test_AUC', 'std_test_Accuracy', 'mean_test_F1',]]

print(cv_df.sort_values('mean_test_AUC', ascending=False))

cv_df.to_excel("C:/Users/Amol Gaikwad/Data Science/employee_attritionsXgb_cv_df.xlsx", index=False)

# =====================================================
#   Important Features
# =====================================================

grid_importance = pd.DataFrame({
    'Feature': X.columns,
    'grid_search_Importance': grid_search.best_estimator_.feature_importances_
})

grid_importance = grid_importance.sort_values(
    by='grid_search_Importance',
    ascending=False
)

print(grid_importance)

grid_importance.to_excel("C:/Users/Amol Gaikwad/Data Science/employee_attritionsXgb _important_features.xlsx", index=False)


output_df = pd.DataFrame({
    'Employee ID': cust_test,
    'Actual_Churn': y_test,
    'GridSearch_Prediction':y_pred,
    'GridSearch_Probability': y_prob
})

print(output_df)

output_df.to_excel("C:/Users/Amol Gaikwad/Data Science/employee_attritionsXgb.xlsx", index=False)


