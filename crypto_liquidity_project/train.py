import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from pathlib import Path

# Load CSVs (assumes they are in same folder as this script)
files = ["../coin_gecko_2022-03-16.csv", "../coin_gecko_2022-03-17.csv"]
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Basic preprocessing
df = df.dropna(subset=['price', '24h_volume', 'mkt_cap']).copy()
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['24h_volume'] = pd.to_numeric(df['24h_volume'], errors='coerce')
df['mkt_cap'] = pd.to_numeric(df['mkt_cap'], errors='coerce')

# Feature engineering: liquidity ratio = 24h_volume / mkt_cap
df['liquidity_ratio'] = df['24h_volume'] / (df['mkt_cap'] + 1e-9)

# Use available numeric features
df['1h'] = pd.to_numeric(df['1h'], errors='coerce')
df['24h'] = pd.to_numeric(df['24h'], errors='coerce')
df['7d'] = pd.to_numeric(df['7d'], errors='coerce')

# Drop rows with NaN in features/target
df = df.dropna(subset=['price','1h','24h','7d','24h_volume','mkt_cap','liquidity_ratio'])

# Define X and y
X = df[['price','1h','24h','7d','24h_volume','mkt_cap']].copy()
y = df['liquidity_ratio']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Test R2: {r2:.6f}")

# Save model and scaler
outdir = Path('.')
pickle.dump(model, open(outdir / 'model.pkl', 'wb'))
pickle.dump(scaler, open(outdir / 'scaler.pkl', 'wb'))

# Save a small sample CSV used for frontend demo
df[['coin','symbol','date','price','1h','24h','7d','24h_volume','mkt_cap','liquidity_ratio']].to_csv(outdir/'processed_sample.csv', index=False)
