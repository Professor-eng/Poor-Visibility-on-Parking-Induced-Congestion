import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

features = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "total_violations"
]
def evaluate(model, X_test, y_test, label):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds)) 
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[{label}] RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def train_ensemble(master_table_path: str):
    df = pd.read_csv(master_table_path)
    X = df[features]
    y = df['aggregated_severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lgb_solo = LGBMRegressor(n_estimators=100, random_state=42)
    lgb_solo.fit(X_train, y_train)
    lgb_metrics = evaluate(lgb_solo, X_test, y_test, "LightGBM solo")

    base_estimators = [
        ('lgb', LGBMRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, max_depth=5, random_state=42))
    ]
    ensemble_model = StackingRegressor(estimators=base_estimators, final_estimator=Ridge())
    ensemble_model.fit(X_train, y_train)
    ens_metrics = evaluate(ensemble_model, X_test, y_test, "Stacked ensemble")

    if ens_metrics["rmse"] < lgb_metrics["rmse"] * 0.97:
        final_model, chosen = ensemble_model, "stacked_ensemble"
        print("Using stacked ensemble.")
    else:
        final_model, chosen = lgb_solo, "lightgbm_solo"
        print("Using LightGBM model.")

    joblib.dump(final_model, "models/ensemble_forecaster.pkl")
    with open("models/training_metrics.json", "w") as f:
        json.dump({"chosen_model": chosen, "lgb_solo": lgb_metrics, "stacked_ensemble": ens_metrics}, f, indent=2)

if __name__ == "__main__":
    train_ensemble("data/master_table.csv")
