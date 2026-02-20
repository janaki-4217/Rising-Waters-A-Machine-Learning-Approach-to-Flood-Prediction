import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Correct dataset path
data_path = os.path.join(os.path.dirname(__file__), "../data/flood_dataset_raw.csv")

df = pd.read_csv(data_path, encoding='latin1')

# Features
X = df[['Temp','Humidity','Cloud Cover','ANNUAL','Jan-Feb',
        'Mar-May','Jun-Sep','Oct-Dec','avgjune','sub']]
y = df['flood']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(X, y)

# Create models folder
model_dir = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "logreg_pipeline.pkl")

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✅ Model trained and saved at: {model_path}")