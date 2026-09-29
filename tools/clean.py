import pandas as pd


def health_score(df: pd.DataFrame) -> int:
    """0-100 data-quality score: penalises missing values, duplicates and mostly-empty columns."""
    miss = df.isna().mean().mean() * 100
    dup = df.duplicated().mean() * 100
    sparse = int((df.isna().mean() > 0.5).sum())
    return int(max(0, round(100 - min(40, miss * 1.5) - min(20, dup * 2) - min(30, sparse * 10))))


def health_label(score: int):
    if score >= 85: return "Excellent", "#10B981"
    if score >= 65: return "Good", "#F59E0B"
    return "Needs cleaning", "#EF4444"


def clean_data(df, drop_cols=(), num_fill="none", cat_fill="none", drop_dups=False):
    """Returns (cleaned copy, log of what was done). Original is never modified."""
    out, log = df.copy(), []
    if drop_cols:
        out = out.drop(columns=list(drop_cols)); log.append(f"Dropped columns: {list(drop_cols)}")
    for c in out.columns:
        if out[c].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            if num_fill in ("median", "mean"):
                val = out[c].median() if num_fill == "median" else out[c].mean()
                out[c] = out[c].fillna(val); log.append(f"{c}: filled with {num_fill}")
        elif cat_fill == "mode":
            out[c] = out[c].fillna(out[c].mode().iloc[0]); log.append(f"{c}: filled with most common value")
    if drop_dups:
        n = int(out.duplicated().sum()); out = out.drop_duplicates(); log.append(f"Removed {n} duplicate rows")
    return out, log
