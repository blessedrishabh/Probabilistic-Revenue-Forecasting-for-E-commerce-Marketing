import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler

class BayesianForecaster:
    def __init__(self):
        self.model = BayesianRidge(compute_score=True)
        self.features = ['log_spend', 'log_clicks', 'log_impressions',
                         'sin_week', 'cos_week', 
                         'lag_1w_revenue', 'lag_2w_revenue', 'lag_4w_revenue', 'lag_52w_revenue',
                         'rolling_4w_avg_ROAS', 'rolling_4w_avg_CVR', 'rolling_4w_avg_CPC',
                         'rolling_8w_avg_revenue', 'rolling_4w_std_revenue',
                         'holiday_flag', 'pre_holiday_week', 'post_holiday_week',
                         'yoy_revenue_growth',
                         'spend_share', 'efficiency_index']
        # Month one-hot features
        self.features += [f'M{m:02d}' for m in range(1, 13)]
        self.scaler = RobustScaler()
        self.max_revenue = 0
        self.train_std = 0
        self.zero_frac = 0  # FIX P2: track zero-inflation
                         
    def fit(self, df):
        if len(df) < 8:
            return False 
        self.max_revenue = df['conversion_value'].max()
        self.train_std = df['conversion_value'].std()
        
        # FIX P2: Track what fraction of training data is zero-revenue
        self.zero_frac = (df['conversion_value'] == 0).mean()
        
        # Use only features that exist in the dataframe
        available_features = [f for f in self.features if f in df.columns]
        self.features_used = available_features
        
        X = df[available_features].fillna(0).astype(float).values
        y = df['log_revenue'].astype(float).values
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model.fit(X_scaled, y)
        return True
        
    def predict(self, X_future, n_samples=5000):
        X = X_future[self.features_used].fillna(0).astype(float).values
        X_scaled = self.scaler.transform(X)
        mu_pred, sigma_pred = self.model.predict(X_scaled, return_std=True)
        
        # FIX P0: Conditional bias correction — only when model is confident
        # For high sigma (volatile/sparse segments), the +0.5*sigma^2 term
        # causes massive over-prediction. Only apply when sigma < 1.0.
        mu_corrected = np.where(
            sigma_pred < 1.0,
            mu_pred + 0.5 * sigma_pred**2,   # standard log-normal correction
            mu_pred                            # no correction for high uncertainty
        )
        
        samples = np.random.normal(mu_corrected, sigma_pred, size=(n_samples, len(mu_pred))).T
        
        # FIX P1: Clip in log-space BEFORE expm1 to prevent overflow
        # exp(25) ≈ 72 billion — way beyond any plausible weekly revenue
        samples = np.clip(samples, a_min=-5, a_max=25)
        rev_samples = np.expm1(samples)
        
        # Clip at 0 on low end and 5x historical max on high end
        if self.max_revenue > 0:
            rev_samples = np.clip(rev_samples, a_min=0, a_max=self.max_revenue * 5)
        else:
            rev_samples = np.maximum(rev_samples, 0)
        
        # FIX P2: Zero-inflated adjustment — if the segment is frequently zero,
        # randomly zero out a matching fraction of the samples (but less aggressive)
        if self.zero_frac > 0.5:
            # Apply a dampened zero rate: only half the historical rate
            effective_zero_rate = self.zero_frac * 0.5
            zero_mask = np.random.random(rev_samples.shape) < effective_zero_rate
            rev_samples[zero_mask] = 0.0
        
        p10 = np.percentile(rev_samples, 10, axis=1)
        p50 = np.percentile(rev_samples, 50, axis=1)
        p90 = np.percentile(rev_samples, 90, axis=1)
        
        # Adaptive uncertainty floor: scale with training data volatility
        for i in range(len(p50)):
            floor_pct = np.clip(sigma_pred[i] * 0.8, 0.15, 0.50)
            min_width = floor_pct * abs(p50[i])
            actual_width = p90[i] - p10[i]
            if actual_width < min_width and p50[i] > 0:
                half_deficit = (min_width - actual_width) / 2
                p90[i] += half_deficit
                p10[i] = max(0, p10[i] - half_deficit)
        
        return p10, p50, p90, rev_samples

class SeasonalNaiveForecaster:
    def __init__(self):
        self.overall_avg = 0
        self.month_avgs = {}
        
    def fit(self, df):
        self.overall_avg = df['conversion_value'].mean()
        df_copy = df.copy()
        df_copy['month'] = df_copy['week_start'].dt.month
        self.month_avgs = df_copy.groupby('month')['conversion_value'].mean().to_dict()
        
    def predict(self, X_future, df_hist):
        last_4w_avg = df_hist['conversion_value'].tail(4).mean()
        last_4w_cv = df_hist['conversion_value'].tail(4).std() / (last_4w_avg + 1e-5)
        last_4w_cv = 0 if pd.isna(last_4w_cv) else last_4w_cv
        
        p10s, p50s, p90s = [], [], []
        rev_samples_list = []
        
        for _, row in X_future.iterrows():
            m = row['week_start'].month
            seasonal_index = self.month_avgs.get(m, self.overall_avg) / (self.overall_avg + 1e-5) if self.overall_avg > 0 else 1.0
            p50 = last_4w_avg * seasonal_index
            
            band = max(0.20, 1.5 * last_4w_cv)
            p10 = p50 * (1 - band)
            p90 = p50 * (1 + band)
            
            p10 = max(0, p10)
            
            p10s.append(p10)
            p50s.append(p50)
            p90s.append(p90)
            
            samples = np.random.normal(p50, (p90-p10)/3.29 + 1e-5, size=5000) 
            samples = np.maximum(samples, 0)
            rev_samples_list.append(samples)
            
        return np.array(p10s), np.array(p50s), np.array(p90s), np.array(rev_samples_list)
