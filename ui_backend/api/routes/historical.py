import os
from fastapi import APIRouter, HTTPException
import pandas as pd

router = APIRouter()

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Forecasting', 'data'))

@router.get("")
def get_historical_data():
    try:
        import sys
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from Forecasting.pipeline import aggregate_to_weekly, load_google_ads, load_meta_ads, load_microsoft_ads
        
        df_g = load_google_ads(os.path.join(DATA_DIR, 'google_ads_campaign_stats.csv'))
        df_m = load_meta_ads(os.path.join(DATA_DIR, 'meta_ads_campaign_stats.csv'))
        df_b = load_microsoft_ads(os.path.join(DATA_DIR, 'bing_campaign_stats.csv'))
        
        df_all = pd.concat([df_g, df_m, df_b], ignore_index=True)
        weekly_df = aggregate_to_weekly(df_all)
        
        # Convert week_start to string
        weekly_df['week_start'] = weekly_df['week_start'].astype(str)
        
        # Sort and return
        weekly_df = weekly_df.sort_values(['week_start', 'channel'])
        return weekly_df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load historical data: {str(e)}")
