import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_demographic_fit(creator_df: pd.DataFrame, campaign_target: dict) -> pd.DataFrame:
    """
    Calculates demographic fit score between each creator and the target audience.
    Uses normalized vector similarity between age/gender/country distributions.
    """
    def vectorize(row, keys):
        return [row.get(k, 0) for k in keys]

    # Normalize and vectorize
    age_keys = ["age_13_17", "age_18_24", "age_25_34", "age_35_44", "age_45_plus"]
    gender_keys = ["male_pct", "female_pct"]
    country_keys = ["US_pct", "MX_pct", "BR_pct", "IN_pct", "CA_pct"]

    def norm(v):
        total = sum(v)
        return [x / total if total > 0 else 0 for x in v]

    creator_df["fit_score"] = 0.0
    for i, row in creator_df.iterrows():
        age_vec = norm(vectorize(row, age_keys))
        gender_vec = norm(vectorize(row, gender_keys))
        country_vec = norm(vectorize(row, country_keys))

        tgt_age = norm([campaign_target[k] for k in age_keys])
        tgt_gender = norm([campaign_target[k] for k in gender_keys])
        tgt_country = norm([campaign_target[k] for k in country_keys])

        sim_age = cosine_similarity([age_vec], [tgt_age])[0][0]
        sim_gender = cosine_similarity([gender_vec], [tgt_gender])[0][0]
        sim_country = cosine_similarity([country_vec], [tgt_country])[0][0]

        creator_df.at[i, "fit_score"] = round(0.4 * sim_age + 0.3 * sim_gender + 0.3 * sim_country, 4)

    return creator_df.sort_values("fit_score", ascending=False)