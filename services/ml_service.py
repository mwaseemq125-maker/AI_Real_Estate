from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "properties.csv"
NUM = ["area_sqft", "bedrooms", "bathrooms", "property_age"]
CAT = ["city", "neighborhood", "property_type"]

def load_data():
    return pd.read_csv(DATA)

def build_model():
    df = load_data()
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), CAT)
    ])
    model = Pipeline([("prep", prep), ("rf", RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2))])
    metrics = {"mae": None, "r2": None}
    X, y = df[NUM + CAT], df["price"]
    if len(df) > 15:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
        test_model = Pipeline([("prep", prep), ("rf", RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2))])
        test_model.fit(Xtr, ytr)
        pred = test_model.predict(Xte)
        metrics = {"mae": round(float(mean_absolute_error(yte, pred))), "r2": round(float(r2_score(yte, pred)), 3)}
    model.fit(X, y)
    return model, metrics

def estimate(payload):
    model, metrics = build_model()
    row = pd.DataFrame([{k: payload[k] for k in NUM + CAT}])
    return {"estimated_price": round(float(model.predict(row)[0])), "metrics": metrics}

def recommend(payload, limit=3):
    df = load_data()
    scale = max(float(df.area_sqft.std()), 1)
    df["match_score"] = (
        (df.city == payload["city"]).astype(float)*3
        + (df.property_type == payload["property_type"]).astype(float)*1.5
        + (df.neighborhood == payload["neighborhood"]).astype(float)
        + (df.bedrooms == int(payload["bedrooms"])).astype(float)
        + (df.bathrooms == int(payload["bathrooms"])).astype(float)*.5
        - (df.area_sqft - float(payload["area_sqft"])).abs()/scale
    )
    return df.sort_values("match_score", ascending=False).head(limit).to_dict("records")

def stats():
    df = load_data()
    return {"total": len(df), "avg_price": round(float(df.price.mean())), "median": round(float(df.price.median())),
            "avg_area": round(float(df.area_sqft.mean())), "city_prices": df.groupby("city").price.mean().round().to_dict(),
            "types": df.property_type.value_counts().to_dict(), "properties": df.to_dict("records")}
