import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

class ResponseCurve:
    def __init__(self):
        self.a = 0
        self.b = 0.75 # Default elasticity

    def fit(self, df):
        df_valid = df[(df['spend'] > 0) & (df['conversion_value'] > 0)]
        if len(df_valid) < 8:
            if len(df_valid) > 0:
                self.a = np.mean(df_valid['conversion_value']) / (np.mean(df_valid['spend'])**self.b + 1e-5)
            return False
            
        y = np.log(df_valid['conversion_value'])
        X = np.log(df_valid['spend'])
        X = add_constant(X)
        
        try:
            model = OLS(y, X).fit()
            if 'spend' in model.params:
                self.b = model.params['spend']
                self.a = np.exp(model.params['const'])
            
            if self.b >= 1.0 or self.b <= 0:
                self.b = 0.75 
                self.a = np.mean(df_valid['conversion_value']) / (np.mean(df_valid['spend'])**self.b + 1e-5)
            return True
        except Exception:
            if len(df_valid) > 0:
                self.a = np.mean(df_valid['conversion_value']) / (np.mean(df_valid['spend'])**self.b + 1e-5)
            return False

    def predict_revenue(self, spend):
        return self.a * (spend ** self.b)

def generate_scenarios(current_spend_pace):
    return {
        'Conservative': 0.8 * current_spend_pace,
        'Baseline': 1.0 * current_spend_pace,
        'Aggressive': 1.2 * current_spend_pace
    }
