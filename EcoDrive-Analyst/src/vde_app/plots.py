import plotly.express as px

def line_power(df):
    fig = px.line(df, x="t", y="P", title="Instantaneous Power")
    return fig
