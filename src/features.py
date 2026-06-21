import pandas as pd

def feature_engineering(indexed_df_path: str) -> pd.DataFrame:
    """
    Transforms ticket-level data into a structured Spatio-Temporal Master Table.
    """
    df = pd.read_csv(indexed_df_path)
    
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='ISO8601', utc=True)
    df['created_datetime'] = df['created_datetime'].dt.tz_convert('Asia/Kolkata')
    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_peak_hour'] = df['hour'].apply(lambda h: 1 if (8 <= h <= 12) or (16 <= h <= 20) else 0)
    
    master_df = df.groupby(['h3_cell', 'hour', 'day_of_week', 'is_weekend', 'is_peak_hour']).agg(
        total_violations=('id', 'count'),
        aggregated_severity=('severity_score', 'sum'),
        mean_latitude=('latitude', 'mean'),
        mean_longitude=('longitude', 'mean')
    ).reset_index()   

    severity_threshold = master_df['aggregated_severity'].quantile(0.75)
    master_df['is_hotspot'] = (master_df['aggregated_severity'] > severity_threshold).astype(int)
    
    return master_df

if __name__ == "__main__":
    master_table = feature_engineering("data/processed_indexed.csv")
    master_table.to_csv("data/master_table.csv", index=False)
    print("Feature engineering succesfully done")