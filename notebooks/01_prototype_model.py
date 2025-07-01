import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # Suppress urllib3/OpenSSL warnings

from sklearn.utils import resample

from src.data_loader import load_raw_data
from src.feature_engineering import engineer_features, extract_caption_embedding
from src.model import train_model
from src.explain import explain_model

# Load data
df = load_raw_data("data/raw/sample_posts.csv")

# Optional: Balance classes if too small
df_majority = df[df["is_viral"] == 1]
df_minority = df[df["is_viral"] == 0]

# Upsample minority to match majority count (for mock data)
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)
df = pd.concat([df_majority, df_minority_upsampled])
df = df.sample(frac=1).reset_index(drop=True)

# Engineer features
df = engineer_features(df)
caption_embeddings = extract_caption_embedding(df["caption"])

# Add embedding features in bulk to avoid fragmentation
emb_df = pd.DataFrame(caption_embeddings, columns=[f'emb_{i}' for i in range(caption_embeddings.shape[1])])
df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

# Define features and target
feature_cols = [col for col in df.columns if col.startswith("emb_") or col in ["caption_length", "has_hashtags"]]
X = df[feature_cols]
y = df["is_viral"]

print("Class distribution:\n", y.value_counts())

# Train model with stratified validation
model, auc = train_model(X, y)
print(f"Validation AUC: {auc:.3f}")

# Generate and save SHAP explanation plot
shap_values = explain_model(model, X)