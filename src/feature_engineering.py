from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_caption_embedding(captions):
    return model.encode(captions.tolist())

def engineer_features(df):
    df['caption_length'] = df['caption'].apply(lambda x: len(str(x)))
    df['has_hashtags'] = df['caption'].apply(lambda x: 1 if '#' in str(x) else 0)
    return df