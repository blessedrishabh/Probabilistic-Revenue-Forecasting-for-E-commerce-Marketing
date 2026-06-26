import pandas as pd

def format_historical_summary(features_df: pd.DataFrame) -> str:
    """
    Formats the last 12 weeks of historical performance into a markdown table.
    """
    if features_df.empty:
        return "No historical data available."
        
    df = features_df.copy()
    
    # We want to group by week, channel, and campaign_type
    if 'week_start' in df.columns:
        df = df.sort_values('week_start', ascending=False)
        
    # Example table format construction
    # Here we simplify the representation for the LLM
    agg = df.groupby(['channel', 'campaign_type'])[['spend', 'conversion_value', 'roas_reported']].mean().reset_index()
    agg.rename(columns={'conversion_value': 'revenue', 'roas_reported': 'roas'}, inplace=True)
    
    md_table = "### Last 12 Weeks Average Performance\n\n"
    md_table += "| Channel | Campaign Type | Avg Weekly Spend | Avg Weekly Revenue | Avg ROAS |\n"
    md_table += "|---|---|---|---|---|\n"
    
    for _, row in agg.iterrows():
        spend = f"${row['spend']:,.2f}"
        rev = f"${row['revenue']:,.2f}"
        roas = f"{row['roas']:.2f}x"
        md_table += f"| {row['channel'].title()} | {row['campaign_type']} | {spend} | {rev} | {roas} |\n"
        
    return md_table
