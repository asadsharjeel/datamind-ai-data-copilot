import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model(df: pd.DataFrame, target: str) -> dict:
    """Mini AutoML: cleans, compares two models with cross-validation, explains the best."""
    if target not in df.columns:
        return {"error": f"Column '{target}' does not exist. Available: {list(df.columns)}"}
    data = df.dropna(subset=[target]).copy()
    X = data.drop(columns=[target])
    y = data[target]

    dropped = [c for c in X.columns
               if X[c].isna().mean() > 0.6 or (X[c].dtype == object and X[c].nunique() > 50)
               or X[c].nunique() == len(X)]
    X = X.drop(columns=dropped)
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

    is_class = y.dtype == object or y.nunique() <= 10
    if is_class:
        task, scoring = "classification", "accuracy"
        models = {"Logistic Regression": LogisticRegression(max_iter=1000),
                  "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)}
    else:
        task, scoring = "regression", "r2"
        models = {"Ridge": Ridge(), "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42)}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if is_class else None)

    scores = {}
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("model", m)])
        scores[name] = float(np.mean(cross_val_score(pipe, X_tr, y_tr, cv=5, scoring=scoring)))
    best = max(scores, key=scores.get)
    final = Pipeline([("pre", pre), ("model", models[best])]).fit(X_tr, y_tr)
    test_score = float(final.score(X_te, y_te))

    imp = permutation_importance(final, X_te, y_te, n_repeats=5, random_state=42)
    importance = (pd.Series(imp.importances_mean, index=X.columns)
                  .sort_values(ascending=False).head(10).round(4))
    return {
        "task": task, "metric": scoring, "cv_scores": {k: round(v, 3) for k, v in scores.items()},
        "best_model": best, "test_score": round(test_score, 3),
        "dropped_columns": dropped, "feature_importance": importance.to_dict(),
        "model": final,
    }
