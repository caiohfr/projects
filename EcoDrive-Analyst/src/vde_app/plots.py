import pandas as pd
import plotly.express as px

def line_power(df: pd.DataFrame):
    fig = px.line(df, x="t", y="P", title="Instantaneous Power")
    return fig

def cycle_chart(df: pd.DataFrame):
    """df deve ter colunas: t (s) e v (m/s)"""
    if not {"t", "v"} <= set(df.columns):
        return None
    dfx = df.copy().dropna(subset=["t", "v"])
    dfx["v_kmh"] =3.6* dfx["v"] # csv ciclos estão em km/h
    fig = px.line(
        dfx, x="t", y="v_kmh",
        labels={"t": "Time [s]", "v_kmh": "Speed [km/h]"},  # <-- labels (minúsculo)
        title="Drive cycle speed profile"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=35, b=0), height=280)
    return fig
