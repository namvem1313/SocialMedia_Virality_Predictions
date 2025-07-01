import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

scaler = StandardScaler()

def train_creator_match_model(df: pd.DataFrame):
    required_features = ['engagement_rate', 'audience_overlap', 'genre_alignment_score', 'follower_count']
    df = df.copy()

    # Ensure all required columns are present
    missing_cols = [col for col in required_features + ['matched'] if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")

    X = df[required_features]
    y = df["matched"]

    if len(set(y)) < 2:
        raise ValueError(f"'matched' column must have both 0 and 1 values. Found only: {set(y)}")

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state=42)

    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc


def score_creators(df: pd.DataFrame, model):
    required_features = ['engagement_rate', 'audience_overlap', 'genre_alignment_score', 'follower_count']
    df = df.copy()

    # Normalize using the same scaler
    X = scaler.transform(df[required_features])
    df["creator_match_score"] = model.predict_proba(X)[:, 1]

    return df
