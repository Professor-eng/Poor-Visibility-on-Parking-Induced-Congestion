import pandas as pd

def feature_engineering(indexed_df_path: str) -> pd.DataFrame:
    df = pd.read_csv(indexed_df_path)
    
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='ISO8601', utc=True)
    df['created_datetime'] = df['created_datetime'].dt.tz_convert('Asia/Kolkata')
    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    df["is_peak_hour"] = (
        ((df["hour"] >= 8) & (df["hour"] <= 12)) |
        ((df["hour"] >= 16) & (df["hour"] <= 20))
    ).astype(int)    
    
    master_df = df.groupby(
        ["h3_cell", "hour", "day_of_week", "is_weekend", "is_peak_hour"]
    ).agg({
        "id": "count",
        "severity_score": "sum",
        "latitude": "mean",
        "longitude": "mean"
    }).reset_index()
    
    severity_threshold = master_df['aggregated_severity'].quantile(0.75)
    master_df['is_hotspot'] = (master_df['aggregated_severity'] > severity_threshold).astype(int)
    
    return master_df

if __name__ == "__main__":
    master_table = feature_engineering("data/processed_indexed.csv")
    master_table.to_csv("data/master_table.csv", index=False)
    print("Feature engineering succesfully done")
