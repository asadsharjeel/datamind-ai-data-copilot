import pandas as pd
import plotly.express as px


def make_chart(df: pd.DataFrame, kind: str, x: str, y: str = None, color: str = None):
    for col in (x, y, color):
        if col and col not in df.columns:
            return None, f"Column '{col}' does not exist. Available: {list(df.columns)}"
    try:
        if kind == "histogram":
            fig = px.histogram(df, x=x, color=color)
        elif kind == "bar":
            data = df.groupby(x)[y].mean().reset_index() if y else df[x].value_counts().reset_index()
            fig = px.bar(data, x=data.columns[0], y=data.columns[1])
        elif kind == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color)
        elif kind == "box":
            fig = px.box(df, x=color, y=x) if color else px.box(df, y=x)
        elif kind == "line":
            fig = px.line(df.sort_values(x), x=x, y=y)
        else:
            return None, "Unsupported chart kind. Use histogram, bar, scatter, box or line."
    except Exception as e:
        return None, f"Chart failed: {e}"
    return fig, "Chart created."
