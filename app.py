import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.ugc_virality import run_virality_prediction
from src.creator_matcher import train_creator_match_model, score_creators
from src.audience_fit import calculate_demographic_fit
from src.trend_timing import forecast_trend_lifecycle
from src.roi_optimizer import optimize_roi
from src.recommender import combine_scores


st.set_page_config(page_title="SocialMedia_Virality_Predictions Dashboard")

st.title("🎬 SocialMedia_Virality_Predictions – AI Powered Influencer Campaign Engine")

# Define tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Creator Matching",
    "📈 UGC Virality Prediction",
    "👥 Audience Fit",
    "📅 Trend Timing Forecast",
    "💰 ROI Optimization",
    "🤖 Final Recommendation Engine"
])

# --------------------------
# Tab 1: Creator Matching
# --------------------------
with tab1:
    st.header("🎯 Creator Matching Module")
    uploaded = st.file_uploader("Upload creator metadata CSV", type=["csv"], key="creator_uploader")

    if uploaded:
        df_creators = pd.read_csv(uploaded)
        st.write("📊 Creator Data Sample:")
        st.dataframe(df_creators.head())

        if "matched" not in df_creators.columns:
            st.warning("Please include a 'matched' column (1 if matched, 0 otherwise) to train the model.")
        else:
            model, auc = train_creator_match_model(df_creators)
            df_scored = score_creators(df_creators, model)

            st.success(f"✅ Model trained with AUC: {auc:.3f}")
            st.subheader("🔝 Top Ranked Creators")
            st.dataframe(df_scored.sort_values("creator_match_score", ascending=False)[["creator_id", "creator_match_score"]])

# --------------------------
# Tab 2: UGC Virality Prediction
# --------------------------
with tab2:
    st.header("📈 UGC Virality Prediction Module")
    uploaded_file = st.file_uploader("Upload creator post CSV", type=["csv"], key="ugc_uploader")

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.subheader("📋 Raw Input Preview")
        st.dataframe(df_input.head())

        df_output, model, shap_fig = run_virality_prediction(df_input)

        st.subheader("📊 Ranked Results")
        st.dataframe(df_output.sort_values("virality_score", ascending=False)[["creator_id", "caption", "virality_score"]])

        st.subheader("🔍 SHAP Feature Importance")
        st.pyplot(shap_fig)

# ---- TAB 3: Audience Fit ----
with tab3:
    st.header("👥 Audience-Demographic Fit")
    uploaded_audience = st.file_uploader("Upload creator audience demographics CSV", type=["csv"], key="audience_uploader")

    if uploaded_audience:
        df_aud = pd.read_csv(uploaded_audience)
        st.write("Uploaded Creator Demographics:")
        st.dataframe(df_aud.head())

        st.subheader("🎯 Enter Campaign Target Demographics")
        with st.form("target_demo_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age_13_17 = st.slider("Age 13–17", 0.0, 1.0, 0.1)
                age_18_24 = st.slider("Age 18–24", 0.0, 1.0, 0.5)
                age_25_34 = st.slider("Age 25–34", 0.0, 1.0, 0.3)
            with col2:
                age_35_44 = st.slider("Age 35–44", 0.0, 1.0, 0.05)
                age_45_plus = st.slider("Age 45+", 0.0, 1.0, 0.05)
                male_pct = st.slider("Male %", 0.0, 1.0, 0.4)
                female_pct = st.slider("Female %", 0.0, 1.0, 0.6)
            with col3:
                US_pct = st.slider("US %", 0.0, 1.0, 0.5)
                MX_pct = st.slider("MX %", 0.0, 1.0, 0.2)
                BR_pct = st.slider("BR %", 0.0, 1.0, 0.1)
                IN_pct = st.slider("IN %", 0.0, 1.0, 0.1)
                CA_pct = st.slider("CA %", 0.0, 1.0, 0.1)

            submitted = st.form_submit_button("Calculate Fit")
            if submitted:
                target_demo = {
                    "age_13_17": age_13_17,
                    "age_18_24": age_18_24,
                    "age_25_34": age_25_34,
                    "age_35_44": age_35_44,
                    "age_45_plus": age_45_plus,
                    "male_pct": male_pct,
                    "female_pct": female_pct,
                    "US_pct": US_pct,
                    "MX_pct": MX_pct,
                    "BR_pct": BR_pct,
                    "IN_pct": IN_pct,
                    "CA_pct": CA_pct
                }
                df_fit = calculate_demographic_fit(df_aud, target_demo)
                st.success("Fit scores calculated!")
                st.dataframe(df_fit[["creator_id", "fit_score"]].sort_values("fit_score", ascending=False))

# ---- TAB 4: Trend Timing & Lifecycle Forecast----
with tab4:
    st.header("📅 Trend Timing & Lifecycle Forecast")
    uploaded_ts = st.file_uploader("Upload trend usage CSV (columns: date, sound_uses)", type=["csv"], key="forecast_uploader")

    if uploaded_ts:
        with st.spinner("Forecasting..."):
            forecast_df, activation_window, fig = forecast_trend_lifecycle(uploaded_ts)

        st.success(f"📌 Suggested Activation Window: **{activation_window}**")
        st.subheader("📈 Forecast Chart")
        st.pyplot(fig)

        with st.expander("📄 Forecast Data (last 10 rows)"):
            st.dataframe(forecast_df.tail(10))

# ---- TAB 5: ROI Optimization & Campaign Learning----
# --- ROI Optimization Tab ---
with tab5:
    st.header("💰 ROI Optimization & Campaign Learning")
    uploaded_campaign = st.file_uploader("Upload campaign data CSV", type=["csv"], key="roi_uploader")

    if uploaded_campaign:
        try:
            df_roi = pd.read_csv(uploaded_campaign)
            from src.roi_optimizer import optimize_roi

            ranked_df, model, mae = optimize_roi(df_roi)
            st.success(f"Model trained! Validation MAE: {mae:.4f}")

            st.subheader("📊 Ranked Creators by Predicted ROI")
            st.dataframe(ranked_df[["creator_id", "predicted_roi", "roi_rank"]].head(10))

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ---- TAB 6: Final Recommendation Engine----
# ---- TAB 6: Final Recommendation ----
with tab6:
    st.header("📁 Final Recommendation Engine")
    st.write("Upload merged scores from all modules to get final ranked creator list.")

    uploaded_final = st.file_uploader("Upload final_recommendation_sample.csv", type=["csv"], key="final_uploader")

    if uploaded_final:
        try:
            df_final = pd.read_csv(uploaded_final)

            required_cols = [
                "creator_id",
                "creator_match_score",
                "fit_score",
                "virality_score",
                "roi_score"
            ]

            # Validate columns
            if not all(col in df_final.columns for col in required_cols):
                st.error(f"Missing one or more required columns. Required: {required_cols}")
            else:
                # Compute final score (you can adjust weights as needed)
                df_final["recommendation_score"] = (
                    0.3 * df_final["creator_match_score"] +
                    0.2 * df_final["fit_score"] +
                    0.3 * df_final["virality_score"] +
                    0.2 * df_final["roi_score"]
                )

                st.success("✅ Recommendation scores computed.")
                st.subheader("🔝 Top Recommended Creators")

                st.dataframe(
                    df_final.sort_values("recommendation_score", ascending=False)[
                        ["creator_id", "recommendation_score"]
                    ]
                )

                csv_download = df_final.sort_values("recommendation_score", ascending=False).to_csv(index=False)
                st.download_button("📥 Download Ranked Results", data=csv_download, file_name="ranked_creators.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error in recommendation module: {str(e)}")
