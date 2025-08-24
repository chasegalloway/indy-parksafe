import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from geopy.distance import geodesic

bundle = joblib.load("model/parksafe_model.pkl")
model = bundle["model"]
FEATURE_COLS = bundle["columns"]

meters = pd.read_csv("data/indy_meters.csv")

def bucket_temp(t):
    if t <= 32: return "frigid"
    if t <= 50: return "cold"
    if t <= 65: return "mild"
    if t <= 80: return "warm"
    return "hot"

def nearest_meter(lat, lng, max_m=120):
    best, best_d = None, float("inf")
    for _, m in meters.iterrows():
        d = geodesic((lat,lng), (m.lat,m.lng)).meters
        if d < best_d:
            best, best_d = m, d
    return (best if best_d <= max_m else None)

def make_features(lat, lng, timestamp, temp_f=None, precip=None, is_holiday=0):
    ts = datetime.fromisoformat(timestamp)
    hour, dow = ts.hour, ts.weekday()
    tb = bucket_temp(temp_f or 70)
    pb = precip if precip in ("dry","precip") else "dry"

    m = nearest_meter(lat, lng)
    assert m is not None, "No meter within 120m"

    base = {
        "hour": hour,
        "day_of_week": dow,
        "is_holiday": int(is_holiday),
        "downtown_flag": int(m.downtown_flag),
        "zone_id": m.zone_id,
        "temp_bucket": tb,
        "precip_bucket": pb
    }

    X = pd.get_dummies(pd.DataFrame([base]), drop_first=True)
    # align columns
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = 0
    return X[FEATURE_COLS]

tests = [
    (39.7683, -86.1586, "2025-06-15T08:30:00", 75, "dry", 0),
    (39.7683, -86.1586, "2025-06-15T17:30:00", 75, "dry", 0),
    (39.7683, -86.1586, "2025-07-04T12:00:00", 85, "dry", 1), 
    (39.7683, -86.1586, "2025-05-10T19:00:00", 70, "precip", 0),
]

for lat, lng, ts, temp, prec, hol in tests:
    X = make_features(lat, lng, ts, temp_f=temp, precip=prec, is_holiday=hol)
    prob = model.predict(X)[0]
    print(f"{ts} | temp={temp}°F precip={prec} holiday={hol} → free prob ≈ {prob:.2%}")
