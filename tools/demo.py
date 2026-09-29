import numpy as np
import pandas as pd

DEMOS = {
    "🚢 Passenger survival (classification)": "titanic",
    "🛒 Retail sales (regression)": "sales",
    "🏠 House prices (regression)": "houses",
}


DEFAULT_TARGET = {"titanic": "Survived", "sales": "Revenue", "houses": "Price"}


def load_demo(key: str) -> pd.DataFrame:
    rng = np.random.default_rng(7); n = 700
    if key == "titanic":
        sex = rng.choice(["male", "female"], n); pc = rng.choice([1, 2, 3], n, p=[.2, .25, .55])
        age = np.clip(rng.normal(30, 13, n), 1, 80).round()
        logit = -0.5 + 2.2 * (sex == "female") - 0.8 * (pc - 2) - 0.02 * (age - 30)
        df = pd.DataFrame(dict(Survived=(rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), Pclass=pc, Sex=sex,
                               Age=age, Fare=(rng.gamma(2, 15, n) * (4 - pc)).round(2),
                               Embarked=rng.choice(["S", "C", "Q"], n, p=[.7, .2, .1]),
                               Cabin=[f"C{i}" if rng.random() < .25 else None for i in range(n)]))
        df.loc[rng.random(n) < .2, "Age"] = np.nan
    elif key == "sales":
        cat = rng.choice(["Electronics", "Clothing", "Home", "Toys"], n); reg = rng.choice(["North", "South", "East", "West"], n)
        disc = rng.choice([0, 5, 10, 20], n); units = rng.integers(1, 30, n)
        price = np.array([{"Electronics": 300, "Clothing": 40, "Home": 90, "Toys": 25}[c] for c in cat]) * rng.uniform(.8, 1.2, n)
        df = pd.DataFrame(dict(Date=pd.date_range("2025-01-01", periods=n, freq="12h").date, Category=cat, Region=reg,
                               Units=units, Discount_pct=disc, Unit_price=price.round(2),
                               Revenue=(units * price * (1 - disc / 100)).round(2)))
        df.loc[rng.random(n) < .05, "Discount_pct"] = np.nan
    else:
        sqft = rng.normal(1600, 500, n).clip(500, 4000).round(); beds = rng.integers(1, 6, n)
        city = rng.choice(["Downtown", "Suburb", "Rural"], n, p=[.3, .5, .2]); age = rng.integers(0, 60, n)
        mult = np.array([{"Downtown": 1.5, "Suburb": 1.0, "Rural": .7}[c] for c in city])
        df = pd.DataFrame(dict(Sqft=sqft, Bedrooms=beds, Location=city, Age_years=age,
                               Price=((sqft * 120 + beds * 8000 - age * 900) * mult + rng.normal(0, 15000, n)).round(-2)))
        df.loc[rng.random(n) < .08, "Age_years"] = np.nan
    return df
