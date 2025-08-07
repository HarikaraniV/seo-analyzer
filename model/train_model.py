import pandas as pd
import lightgbm as lgb
import joblib
import os

# Construct absolute path to dataset and model
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "seo_dataset.csv"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "seo_model.pkl"))

# 1. Load dataset
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset with shape: {df.shape}")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    exit(1)

# 2. Feature-target split
X = df.drop("SEO_Score", axis=1)
y = df["SEO_Score"]

# 3. Train LightGBM model
print("🚀 Training LightGBM model...")
model = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X, y)
print("✅ Model training completed!")

# 4. Save model
joblib.dump(model, MODEL_PATH)
print(f"📦 Model saved to: {MODEL_PATH}")
