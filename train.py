import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

shard_paths = sorted(glob.glob(os.path.join('data', 'indy_parksafe_synth_shard_*.csv')))
print(f"Loading {len(shard_paths)} shards...")
dfs = [pd.read_csv(path, parse_dates=['timestamp']) for path in shard_paths]
df = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset shape: {df.shape}")

if 'temp_bucket' not in df.columns:
    df['temp_bucket'] = pd.cut(df['temp'], bins=[-50, 32, 50, 65, 80, 150], labels=['frigid','cold','mild','warm','hot'])
if 'precip_bucket' not in df.columns:
    df['precip_bucket'] = df['precip'].map({0:'dry', 1:'precip'})

features = ['hour', 'day_of_week', 'is_holiday', 'zone_id', 'downtown_flag', 'temp_bucket', 'precip_bucket']
target = 'label_free_prob'

X = pd.get_dummies(df[features], drop_first=True)
y = df[target]
print(f"Feature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples.")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)
print("Training RandomForestRegressor...")
model.fit(X_train, y_train)

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"Evaluation -> R^2: {r2:.4f}, MAE: {mae:.4f}")

os.makedirs('model', exist_ok=True)
joblib.dump({'model': model, 'columns': X.columns.tolist()}, os.path.join('model', 'parksafe_model.pkl'))
print("Model and columns saved to model/parksafe_model.pkl")

