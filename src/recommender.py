import pandas as pd

def combine_scores(df: pd.DataFrame,
                   weights: dict = {
                       "creator_match_score": 0.3,
                       "fit_score": 0.2,
                       "virality_score": 0.3,
                       "roi_score": 0.2
                   }) -> pd.DataFrame:
    # Ensure all required scores are present
    for key in weights:
        if key not in df.columns:
            raise ValueError(f"Missing required column: {key}")

    # Normalize scores
    for key in weights:
        min_val, max_val = df[key].min(), df[key].max()
        if max_val != min_val:
            df[f"{key}_norm"] = (df[key] - min_val) / (max_val - min_val)
        else:
            df[f"{key}_norm"] = 0.5  # fallback if constant

    # Weighted sum for recommendation score
    df["recommendation_score"] = sum(
        df[f"{key}_norm"] * weight for key, weight in weights.items()
    )

    return df.sort_values("recommendation_score", ascending=False)
