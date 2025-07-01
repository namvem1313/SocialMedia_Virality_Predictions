import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

REQUIRED_COLUMNS = [
    "creator_id", "creator_cost", "audience_reach", "engagement_rate",
    "ugc_generated", "campaign_type", "region", "content_type"
]

def optimize_roi(df: pd.DataFrame):
    # --- Check all required columns ---
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing one or more required columns for ROI optimization: {missing}")

    df = df.copy()

    # --- Compute ROI ---
    df["roi"] = df["ugc_generated"] / df["creator_cost"].replace(0, 1)  # avoid divide by zero

    # --- Prepare features ---
    X = df[["audience_reach", "engagement_rate", "creator_cost", "campaign_type", "region", "content_type"]]
    y = df["roi"]

    # --- One-hot encoding for categorical features ---
    X_encoded = pd.get_dummies(X, columns=["campaign_type", "region", "content_type"], drop_first=True)

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # --- Train model ---
    model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # --- Predict ROI ---
    df["predicted_roi"] = model.predict(X_encoded)
    df["roi_rank"] = df["predicted_roi"].rank(ascending=False)

    # --- Score ---
    mae = mean_absolute_error(y_test, model.predict(X_test))

    return df.sort_values("roi_rank"), model, mae
