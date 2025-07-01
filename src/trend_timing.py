import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def forecast_trend_lifecycle(file):
    df = pd.read_csv(file)
    df.rename(columns={"date": "ds", "sound_uses": "y"}, inplace=True)

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # Activation window rule (example heuristic)
    peak_day = forecast["yhat"].idxmax()
    today = len(df)
    window = "Early" if today < peak_day - 2 else "Peak" if abs(today - peak_day) <= 2 else "Late"

    # Plotting
    fig = model.plot(forecast)
    plt.title("Trend Lifecycle Forecast")

    return forecast, window, fig