import os
import json
import logging
from datetime import timedelta
import pandas as pd
import numpy as np
import pickle
from pipeline import load_google_ads, load_meta_ads, load_microsoft_ads, aggregate_to_weekly
from feature_engineering import compute_features
from models import BayesianForecaster, SeasonalNaiveForecaster
from budget_simulator import ResponseCurve, generate_scenarios

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
def train_model():
        logging.info("Ingesting raw CSVs...")
        df_google = load_google_ads('data/google_ads_campaign_stats.csv')
        df_meta = load_meta_ads('data/meta_ads_campaign_stats.csv')
        df_ms = load_microsoft_ads('data/bing_campaign_stats.csv')

        df_all = pd.concat([df_google, df_meta, df_ms], ignore_index=True)
        if df_all.empty:
            logging.error("No data loaded. Check file paths.")
            return

        logging.info("Schema normalization & weekly aggregation...")
        weekly_df = aggregate_to_weekly(df_all)

        logging.info("Feature engineering...")
        features_df = compute_features(weekly_df)

        logging.info("Model fitting and response curve estimation...")
        segments = features_df.groupby(['channel', 'campaign_type'])
        models = {}
        curves = {}

        for (ch, ct), grp in segments:
            bf = BayesianForecaster()
            if not bf.fit(grp):
                snf = SeasonalNaiveForecaster()
                snf.fit(grp)
                models[(ch, ct)] = ('snf', snf, grp)
            else:
                models[(ch, ct)] = ('bayesian', bf, grp)

            rc = ResponseCurve()
            rc.fit(grp)
            curves[(ch, ct)] = rc
        with open ('pickle/model.pkl', 'wb') as f:
            pickle.dump({'models': models, 'curves' : curves}, f)
            logging.info("Model successfully saved to pickle/model.pkl")

if __name__ == "__main__":
    train_model()