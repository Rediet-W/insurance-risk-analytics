# src/modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def preprocess_and_split_regression(df, target_col='TotalClaims', test_size=0.2, random_state=42):
    """
    Prepares a clean subset of data where a claim occurred (Claims > 0)
    for Severity Regression Modeling, with bulletproof type handling.
    """
    # Filter for rows where a claim actually happened
    sub_df = df[df[target_col] > 0].copy()
    
    # 1. Feature Engineering: Create cleanly parsed numeric columns
    if 'RegistrationYear' in sub_df.columns:
        sub_df['VehicleAge'] = 2015 - pd.to_numeric(sub_df['RegistrationYear'], errors='coerce')
    else:
        sub_df['VehicleAge'] = 0
        
    if 'CustomValueEstimate' in sub_df.columns:
        sub_df['CustomValueEstimate'] = pd.to_numeric(sub_df['CustomValueEstimate'], errors='coerce')

    # Define the core feature set to feed into the algorithms
    feature_cols = ['Province', 'Gender', 'VehicleType', 'CustomValueEstimate', 'VehicleAge']
    feature_cols = [col for col in feature_cols if col in sub_df.columns]
    
    X = sub_df[feature_cols].copy()
    y = sub_df[target_col].copy()
    
    # 2. Strict type splitting loop
    for col in X.columns:
        if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.StringDtype) or str(X[col].dtype) == 'category':
            X[col] = X[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)
            
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_regression_models(X_train, X_test, y_train, y_test):
    """
    Trains and compares Linear Regression, Random Forest, and XGBoost Regressors.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    metrics_log = {}
    for name, model in models.items():
        # Fit model on training data
        model.fit(X_train, y_train)
        # Generate predictions on test data
        preds = model.predict(X_test)
        
        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        metrics_log[name] = {"RMSE": round(rmse, 2), "R2": round(r2, 4), "model_obj": model}
        
    return metrics_log

    # Append to the bottom of src/modeling.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

def preprocess_and_split_classification(df, target_flag='HasClaim', test_size=0.2, random_state=42):
    """
    Prepares the entire dataset to predict the likelihood of a claim occurring (0 or 1).
    """
    df_copy = df.copy()
    
    # Target flag generation
    df_copy[target_flag] = np.where(df_copy['TotalClaims'] > 0, 1, 0)
    
    # Feature Engineering
    if 'RegistrationYear' in df_copy.columns:
        df_copy['VehicleAge'] = 2015 - pd.to_numeric(df_copy['RegistrationYear'], errors='coerce')
    else:
        df_copy['VehicleAge'] = 0
        
    if 'CustomValueEstimate' in df_copy.columns:
        df_copy['CustomValueEstimate'] = pd.to_numeric(df_copy['CustomValueEstimate'], errors='coerce')

    feature_cols = ['Province', 'Gender', 'VehicleType', 'CustomValueEstimate', 'VehicleAge']
    feature_cols = [col for col in feature_cols if col in df_copy.columns]
    
    X = df_copy[feature_cols].copy()
    y = df_copy[target_flag].copy()
    
    for col in X.columns:
        if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.StringDtype) or str(X[col].dtype) == 'category':
            X[col] = X[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)
            
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def train_and_evaluate_classifier(X_train, X_test, y_train, y_test):
    """
    Trains an XGBoost Classifier with explicit scale adjustments 
    to handle highly imbalanced insurance data.
    """
    # Calculate the exact ratio of non-claims to claims to balance the classes
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_factor = num_neg / num_pos if num_pos > 0 else 1.0
    
    # Initialize the classifier using the calculated scale factor
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_factor,  # Adjusts weights for the rare claim class
        random_state=42,
        eval_metric='logloss'
    )
    
    clf.fit(X_train, y_train)
    
    # 1. Hard Predictions for metrics evaluation
    preds = clf.predict(X_test)
    
    # 2. Extract continuous probabilities for the pricing formula
    # column [:, 1] represents the probability of a claim occurring: P(claim)
    probs = clf.predict_proba(X_test)[:, 1]
    
    return {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, preds, zero_division=0), 4),
        "probabilities": probs,
        "model_obj": clf
    }