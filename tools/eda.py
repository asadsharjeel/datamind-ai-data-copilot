import pandas as pd


def quick_eda(df: pd.DataFrame) -> dict:
    """Automatic data-quality report. Returns a dict so the UI, the LLM or a file can use it."""
    missing = df.isna().sum()
    num = df.select_dtypes(include="number")
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_types": df.dtypes.astype(str).to_dict(),
        "missing_values": missing.to_dict(),
        "missing_percent": (df.isna().mean() * 100).round(1).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": int(num.shape[1]),
        "most_missing_column": missing.idxmax(),
        "high_correlations": _top_corr(num),
        "outlier_counts": _outliers(num),
    }


def _top_corr(num, threshold=0.7):
    if num.shape[1] < 2:
        return []
    c = num.corr().abs()
    out = []
    cols = list(c.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if c.iloc[i, j] >= threshold:
                out.append((cols[i], cols[j], round(float(c.iloc[i, j]), 2)))
    return sorted(out, key=lambda t: -t[2])[:10]


def _outliers(num):
    res = {}
    for col in num.columns:
        q1, q3 = num[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((num[col] < q1 - 1.5 * iqr) | (num[col] > q3 + 1.5 * iqr)).sum())
        if n:
            res[col] = n
    return res
