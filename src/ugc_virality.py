import pandas as pd
from .feature_engineering import engineer_features, extract_caption_embedding
from .model import train_model
from .explain import explain_model
import matplotlib.pyplot as plt

def run_virality_prediction(df: pd.DataFrame):
    # Step 1: Feature engineering
    df = engineer_features(df)
    caption_embeddings = extract_caption_embedding(df["caption"])
    for i in range(caption_embeddings.shape[1]):
        df[f"emb_{i}"] = caption_embeddings[:, i]

    # Step 2: Inject pseudo-labels if needed
    if "is_viral" not in df.columns:
        df["is_viral"] = df["caption"].apply(
            lambda x: 1 if any(kw in x.lower() for kw in ["viral", "trend", "must watch", "can't believe"]) else 0
        )

    # Step 3: Train model
    X = df[[col for col in df.columns if col.startswith("emb_") or col in ["caption_length", "has_hashtags"]]]
    y = df["is_viral"]
    model, auc = train_model(X, y)

    # Step 4: Predict
    df["virality_score"] = model.predict_proba(X)[:, 1]

    # Step 5: SHAP
    shap_values = explain_model(model, X)
    fig = plt.gcf()

    return df, model, fig