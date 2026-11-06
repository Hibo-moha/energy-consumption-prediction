
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./dataset/clean_energy.csv"

df = pd.read_csv(DATA_PATH)

# 1. Separate features and target
X = df.drop(columns=["EnergyConsumption"])
y = df["EnergyConsumption"]

# 2. Handle missing values
X = X.copy()

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

for col in X.select_dtypes(include=["number"]).columns:
    X[col] = X[col].fillna(X[col].median())

# 3. Encode categorical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Scale numerical features for Linear Regression
scaler = StandardScaler()

X_train_lr = scaler.fit_transform(X_train)
X_test_lr = scaler.transform(X_test)

# 6. Train Linear Regression
lr = LinearRegression()
lr.fit(X_train_lr, y_train)

# 7. Train Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

# 8. Save models and preprocessing objects
os.makedirs("./models", exist_ok=True)

joblib.dump(lr, "./models/linear_regression.pkl")
joblib.dump(rf, "./models/random_forest.pkl")
joblib.dump(scaler, "./models/scaler.pkl")
joblib.dump(X.columns.tolist(), "./models/feature_columns.pkl")

print("Training complete!")
print("Linear Regression trained successfully.")
print("Random Forest trained successfully.")
print("Models and preprocessing objects saved.")
