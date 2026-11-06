
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

# 1. Load dataset
DATA_PATH = "./dataset/clean_energy.csv"
df = pd.read_csv(DATA_PATH)

# 2. Separate features and target
X = df.drop(columns=["EnergyConsumption"])
y = df["EnergyConsumption"]

# 3. Handle missing values (same approach as training)
for col in X.select_dtypes(include=["object", "str"]).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

for col in X.select_dtypes(include=["number"]).columns:
    X[col] = X[col].fillna(X[col].median())

# 4. Encode categorical features
X = pd.get_dummies(X, drop_first=True, dtype=int)

# 5. Train/test split (same random state)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Load saved models and preprocessing objects
lr = joblib.load("./models/linear_regression.pkl")
rf = joblib.load("./models/random_forest.pkl")
scaler = joblib.load("./models/scaler.pkl")

feature_columns = joblib.load(
    "./models/feature_columns.pkl"
)

# 7. Align test features
X_test = X_test.reindex(
    columns=feature_columns,
    fill_value=0
)

# 8. Scale Linear Regression input
X_test_lr = scaler.transform(X_test)

# 9. Generate predictions
p_lr = lr.predict(X_test_lr)
p_rf = rf.predict(X_test)

# 10. Evaluation function
def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n========== {name} ==========")
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# 11. Evaluate both models
lr_results = evaluate_model(
    "LINEAR REGRESSION",
    y_test,
    p_lr
)

rf_results = evaluate_model(
    "RANDOM FOREST",
    y_test,
    p_rf
)

# 12. Compare models
results = pd.DataFrame([
    lr_results,
    rf_results
])

print("\n========== MODEL COMPARISON ==========")
print(results.to_string(index=False))

best_model = results.loc[
    results["R2"].idxmax()
]

print("\nBest Model:", best_model["Model"])
print("Best R² Score:", round(best_model["R2"], 4))

# 13. Single-row sanity check
i = 3

x_one_df = X_test.iloc[[i]]
y_true = y_test.iloc[i]

x_one_lr = scaler.transform(x_one_df)

p_lr_one = float(lr.predict(x_one_lr)[0])
p_rf_one = float(rf.predict(x_one_df)[0])

print("\n========== SINGLE ROW CHECK ==========")
print(f"Actual Energy: {y_true:.2f}")
print(f"LR Prediction: {p_lr_one:.2f}")
print(f"RF Prediction: {p_rf_one:.2f}")

# 14. Create output directory
os.makedirs("./outputs", exist_ok=True)

# 15. Actual vs Predicted plot
plt.figure()
plt.scatter(y_test, p_rf)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted - Random Forest")
plt.tight_layout()
plt.savefig("./outputs/actual_vs_predicted.png")
plt.close()

# 16. Residual plot
residuals = y_test - p_rf

plt.figure()
plt.scatter(p_rf, residuals)
plt.axhline(0)
plt.xlabel("Predicted Energy Consumption")
plt.ylabel("Residuals")
plt.title("Residual Plot - Random Forest")
plt.tight_layout()
plt.savefig("./outputs/residuals.png")
plt.close()

print("\nEvaluation completed successfully!")
print("Charts saved inside outputs/ folder.")
