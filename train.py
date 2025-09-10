import glob
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from mlon import DataPreprocessor, ModelEvaluator, Visualizer, ModelUtils, LeakageDetector
from mlon.guardrails import BiasDetector
from sklearn.model_selection import train_test_split

preprocessor = DataPreprocessor()
evaluator = ModelEvaluator()
visualizer = Visualizer()
model_utils = ModelUtils()
leakage_detector = LeakageDetector()

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

X = preprocessor.encode_categorical(df[features], method='onehot')
y = df[target]
print(f"Feature matrix shape: {X.shape}")

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"Training on {train_x.shape[0]} samples, testing on {test_x.shape[0]} samples.")

leakage_warnings = leakage_detector.check_train_test_overlap(train_x, test_x)
if leakage_warnings:
    print("\nData Leakage Warnings:")
    for warning_type, warnings in leakage_warnings.items():
        for warning in warnings:
            print(f"- {warning}")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)
print("\nTraining RandomForestRegressor...")
model.fit(train_x, train_y)

predictions = model.predict(test_x)
metrics = evaluator.regression_metrics(test_y, predictions)
print("\nModel Performance:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

feature_importance = model.feature_importances_
visualizer.plot_feature_importance(
    importance=feature_importance,
    feature_names=X.columns.tolist(),
    top_n=10
)

bias_detector = BiasDetector()
if 'zone_id' in df.columns:
    bias_warnings = bias_detector.check_dataset_bias(
        df, protected_features=['zone_id']
    )
    if bias_warnings:
        for warning_type, warnings in bias_warnings.items():
            for warning in warnings:
                print(f"- {warning}")

os.makedirs('model', exist_ok=True)
model_utils.save_model(
    {'model': model, 'columns': X.columns.tolist()},
    os.path.join('model', 'parksafe_model.pkl')
)
print("\nModel and columns saved to model/parksafe_model.pkl")

