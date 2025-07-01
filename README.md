Author - Lakshmi Namratha Vempaty

Project Overview
Influur Pulse is a modular Streamlit dashboard for influencer marketing campaign planning, designed to recommend optimal TikTok creators for music or brand campaigns using predictive AI and campaign analytics.

Folder Structure
bash
Copy
Edit
influur_pulse/
│
├── app.py                       # Streamlit app
├── requirements.txt             # Python dependencies
├── README.md
│
├── data/
│   ├── raw/                     # Raw input samples (CSV uploads)
│   └── processed/               # Processed/staged outputs (optional)
│
├── notebooks/                   # Prototyping & experimentation notebooks
├── src/                         # Modular pipeline code
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── explain.py
│   ├── creator_matcher.py
│   ├── audience_fit.py
│   ├── trend_timing.py
│   ├── roi_optimizer.py
│   └── recommendation.py

Local Setup

Clone repository

    git clone https://github.com/your_username/influur_pulse.git
    cd influur_pulse

Create and activate virtual environment

    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    .venv\Scripts\activate     # Windows

Install dependencies

    pip install -r requirements.txt

Run Streamlit app

    streamlit run app.py

Sample CSVs to Upload

| Tab                  | File to Upload                     | Sample Path                                 |
| -------------------- | ---------------------------------- | ------------------------------------------- |
| Creator Matching     | `creator_metadata_sample.csv`      | `data/raw/Tab1.csv`      |
| UGC Virality         | `ugc_caption_sample.csv`           | `data/raw/Tab2csv`           |
| Audience Fit         | `audience_demographics_sample.csv` | `data/raw/Tab3.csv` |
| Trend Timing         | `sound_trend_sample.csv`           | `data/raw/Tab4csv`           |
| ROI Optimization     | `campaign_sample.csv`              | `data/raw/Tab5.csv`        |
| Final Recommendation | `final_recommendation_sample.csv`  | `data/raw/Tab6.csv`  |


Tab-by-Tab Explanation & Data Flow

1️⃣ Tab: Creator Matching

    Input: creator_metadata_sample.csv
    What It Does:
        Trains a model to predict if a creator is a good fit for campaigns.
        Adds creator_match_score (0 to 1).
    Passes To: Final recommendation as one of the weighted inputs.

2️⃣ Tab: UGC Virality Prediction

    Input: ugc_caption_sample.csv
    What It Does:
        Uses caption content & embeddings to predict likelihood of virality.
        Adds virality_score.
    Passes To: Final recommendation.

3️⃣ Tab: Audience Fit

    Input: audience_demographics_sample.csv + slider for campaign target demographics
    What It Does:
        Compares creator audience with campaign targets.
        Computes a fit_score between 0 and 1.
    Passes To: Final recommendation.

4️⃣ Tab: Trend Timing & Lifecycle Forecast

    Input: sound_trend_sample.csv (date, sound_uses)
    What It Does:
        Uses time-series modeling to identify whether a sound is in Early, Peak, or Late stage.
    Standalone: Informs content launch timing.

5️⃣ Tab: ROI Optimization & Campaign Learning

    Input: campaign_sample.csv
    What It Does:
        Calculates roi_score based on cost, region, past performance.
    Passes To: Final recommendation.

6️⃣ Tab: Final Recommendation Engine

    Input: final_recommendation_sample.csv (merged scores from above modules)
    What It Does:
        Computes weighted recommendation_score from:
        creator_match_score
        fit_score
        virality_score
        roi_score
    Outputs ranked creators and downloadable CSV.


Project Output: 

Each tab in the Influur Pulse dashboard contributes a unique predictive signal to evaluate and rank creators for influencer campaigns. The Creator Matching tab predicts alignment between a creator’s profile and campaign needs, outputting a creator_match_score. The UGC Virality Prediction tab assesses the viral potential of content using caption analysis and returns a virality_score. The Audience Fit tab measures how closely a creator's audience matches target demographics, producing a fit_score. The Trend Timing module identifies the ideal phase (e.g., Early, Peak) to launch content based on usage trajectories. The ROI Optimization tab estimates content efficiency and cost-effectiveness, generating an roi_score. Finally, the Recommendation Engine combines all these outputs into a unified recommendation_score, enabling marketers to prioritize creators who are well-matched, demographically aligned, likely to go viral, cost-efficient, and timed for maximum trend impact.