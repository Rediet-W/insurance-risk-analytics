import pandas as pd
from src.eda_utils import calculate_loss_ratio

def test_calculate_loss_ratio_normal():
    """Test standard loss ratio calculations."""
    data = pd.DataFrame({
        'TotalClaims': [100, 200],
        'TotalPremium': [200, 200]
    })
    # Total claims = 300, Total premium = 400 -> 300/400 = 0.75
    assert calculate_loss_ratio(data) == 0.75

def test_calculate_loss_ratio_zero_premium():
    """Test that zero premium returns 0 instead of causing a ZeroDivisionError."""
    data = pd.DataFrame({
        'TotalClaims': [100],
        'TotalPremium': [0]
    })
    assert calculate_loss_ratio(data) == 0