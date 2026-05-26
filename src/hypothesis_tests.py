# src/hypothesis_tests.py
import pandas as pd
import numpy as np
from scipy import stats

def run_categorical_frequency_test(df, group_col, category_a, category_b, claim_flag_col='HasClaim'):
    """
    Runs a Chi-Squared test of independence to check if claim frequency (proportion 
    of policies with at least one claim) differs significantly between Group A and Group B.
    """
    # Filter groups
    group_a_data = df[df[group_col] == category_a]
    group_b_data = df[df[group_col] == category_b]
    
    # Create contingency table
    a_claims = group_a_data[claim_flag_col].sum()
    a_no_claims = len(group_a_data) - a_claims
    
    b_claims = group_b_data[claim_flag_col].sum()
    b_no_claims = len(group_b_data) - b_claims
    
    contingency_table = [[a_claims, a_no_claims], [b_claims, b_no_claims]]
    
    # Handle edge case where no claims exist in a group
    if min(a_claims, b_claims) == 0:
        return 0.0, 1.0, "Insufficient data for chi-squared check"
        
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return chi2, p_value

def run_numerical_t_test(df, group_col, category_a, category_b, numerical_kpi_col):
    """
    Runs a two-sample independent t-test to check if numerical averages 
    (Claim Severity or Margin) differ significantly between Group A and Group B.
    """
    # Extract arrays while dropping missing values
    group_a_metrics = df[df[group_col] == category_a][numerical_kpi_col].dropna()
    group_b_metrics = df[df[group_col] == category_b][numerical_kpi_col].dropna()
    
    # Run independent t-test (equal_var=False handles unequal sample sizes and variances safely)
    t_stat, p_value = stats.ttest_ind(group_a_metrics, group_b_metrics, equal_var=False)
    return t_stat, p_value