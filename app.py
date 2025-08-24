# app.py
# FastAPI service with user-friendly, styled UI: input address + radius, outputs free-spot probabilities for nearby meters.

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
from geopy.distance import geodesic
import os

app = FastAPI(title="Indy ParkSafe Web App")

BASE_DIR = os.path.dirname(__file__)
bundle = joblib.load(os.path.join(BASE_DIR, 'model', 'parksafe_model.pkl'))
model = bundle['model']
FEATURE_COLS = bundle['columns']
meters = pd.read_csv(os.path.join(BASE_DIR, 'data', 'indy_meters.csv'))

geolocator = Nominatim(user_agent="indy_parksafe_app", timeout=10)

def bucket_temp(t):
    if t <= 32: return "frigid"
    if t <= 50: return "cold"
    if t <= 65: return "mild"
    if t <= 80: return "warm"
    return "hot"

def prepare_features(m, ts: datetime, temp_f: float, is_holiday: int):
    hour, dow = ts.hour, ts.weekday()
    tb = bucket_temp(temp_f)
    pb = "dry"

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
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = 0
    return X[FEATURE_COLS]

STYLE = """
<style>
  body { font-family: Arial, sans-serif; background-color: #f4f6f8; margin: 0; padding: 0; }
  .container { max-width: 800px; margin: 40px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  h1, h2 { color: #333; }
  form label { display: block; margin: 10px 0 5px; color: #555; }
  form input, form select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
  .checkbox-label { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
  form button { margin-top: 15px; padding: 10px 20px; background-color: #007acc; color: white; border: none; border-radius: 4px; cursor: pointer; }
  form button:hover { background-color: #005fa3; }
  table { width: 100%; border-collapse: collapse; margin-top: 20px; }
  table th, table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
  .error { color: red; margin-top: 20px; }
  .back-link { display: inline-block; margin-top: 20px; color: #007acc; text-decoration: none; }
  .back-link:hover { text-decoration: underline; }
  .tooltip {
  position: fixed;
  background: rgba(0,0,0,0.8);
  color: #fff;
  padding: 6px 8px;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
  z-index: 1000;
  display: none;
  }
</style>
"""

@app.get("/", response_class=HTMLResponse)
def form_page():
    return HTMLResponse(f"""
    <html><head><title>Indy ParkSafe</title>{STYLE}</head><body>
      <div class="container">
        <h1>Indy ParkSafe</h1>
        <h2>Estimate Free Parking Probability</h2>
        <form method="post" action="/predict">
          <label>Address</label>
          <input type="text" name="address" placeholder="123 Main St, Indianapolis, IN" required>

          <label>Radius (meters)</label>
          <input type="number" name="radius" value="200" min="50" required>

          <label>Date & Time</label>
          <input type="datetime-local" name="datetime" required>

          <label>Temperature (°F)</label>
          <input type="number" name="temp_f" value="70">

          <label class="checkbox-label">Holiday <input type="checkbox" name="is_holiday" value="1"></label>

          <button type="submit">Estimate Availability</button>
        </form>
      </div>
    </body></html>
    """)

@app.post("/predict", response_class=HTMLResponse)
def predict_address(
    address: str = Form(...),
    radius: float = Form(...),
    datetime: str = Form(...),
    temp_f: float = Form(70.0),
    is_holiday: str = Form(None)
):
    try:
        location = geolocator.geocode(address)
    except GeocoderUnavailable:
        return HTMLResponse(f"<div class='container'><p class='error'>Geocoding service unavailable. Please try again later.</p><a class='back-link' href='/'>New Query</a></div>")
    if not location:
        return HTMLResponse(f"<div class='container'><p class='error'>Could not geocode address: {address}</p><a class='back-link' href='/'>New Query</a></div>")

    try:
        ts = pd.to_datetime(datetime)
        ts = ts.replace(year=2024)
    except:
        return HTMLResponse(f"<div class='container'><p class='error'>Invalid date/time format.</p><a class='back-link' href='/'>New Query</a></div>")

    holiday_flag = 1 if is_holiday else 0

    meters['distance'] = meters.apply(lambda row: geodesic((location.latitude, location.longitude), (row.lat, row.lng)).meters, axis=1)
    nearby = meters[meters['distance'] <= radius].copy()
    if nearby.empty:
        return HTMLResponse(f"<div class='container'><p class='error'>No meters found within {radius}m of {address}.</p><a class='back-link' href='/'>New Query</a></div>")

    results = []
    for _, m in nearby.iterrows():
        X = prepare_features(m, ts, temp_f, holiday_flag)
        prob = model.predict(X)[0]
        results.append((m.meter_id, m.distance, prob, m.lat, m.lng))

    results.sort(key=lambda x: x[1])
    avg_prob = np.mean([p for _, _, p, _, _ in results])

    rows = "".join(
    f"<tr class='meter-row' data-lat='{lat}' data-lng='{lng}'>"
    f"<td>{mid}</td><td>{dist:.1f} m</td><td>{prob*100:.1f}%</td></tr>"
    for mid, dist, prob, lat, lng in results
)

    return HTMLResponse(f"""
    <html><head><title>Results - Indy ParkSafe</title>{STYLE}</head><body>
      <div class="container">
        <div id="tooltip" class="tooltip"></div>
        <h1>Availability near {address}</h1>
        <p>Found <strong>{len(results)}</strong> meters within <strong>{radius}m</strong>.</p>
        <p><strong>Average free chance:</strong> {avg_prob*100:.1f}%</p>
        <table>
          <thead><tr><th>Meter ID</th><th>Distance</th><th>Free %</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <a class="back-link" href="/">New Query</a>
      </div>
      <script>
  const tooltip = document.getElementById('tooltip');

  function showTip(e, text) {{
    tooltip.textContent = text;
    tooltip.style.display = 'block';
    const offset = 12;
    tooltip.style.left = (e.clientX + offset) + 'px';
    tooltip.style.top  = (e.clientY + offset) + 'px';
  }}

  function hideTip() {{
    tooltip.style.display = 'none';
  }}

  // Delegate hover events to table rows with class 'meter-row'
  document.addEventListener('mouseover', (e) => {{
    const row = e.target.closest('.meter-row');
    if (!row) return;
    const lat = row.getAttribute('data-lat');
    const lng = row.getAttribute('data-lng');
    showTip(e, `lat: ${{Number(lat).toFixed(6)}}, lng: ${{Number(lng).toFixed(6)}}`);
  }});

  document.addEventListener('mousemove', (e) => {{
    if (tooltip.style.display === 'block') {{
      const offset = 12;
      tooltip.style.left = (e.clientX + offset) + 'px';
      tooltip.style.top  = (e.clientY + offset) + 'px';
    }}
  }});

  document.addEventListener('mouseout', (e) => {{
    const row = e.target.closest('.meter-row');
    if (!row) return;
    hideTip();
  }});
</script>

    </body></html>
    """)
