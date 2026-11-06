
import joblib
import pandas as pd

# Electricity price
price_per_kwh = 0.50
currency = "USD"

# 1. Load saved models and preprocessing objects
lr = joblib.load("./models/linear_regression.pkl")
rf = joblib.load("./models/random_forest.pkl")
scaler = joblib.load("./models/scaler.pkl")
feature_columns = joblib.load("./models/feature_columns.pkl")

# 2. New energy usage data
temperature = 25.0
humidity = 45.0
square_footage = 1500.0
occupancy = 5
hvac_usage = "On"
lighting_usage = "On"
renewable_energy = 15.0
day_of_week = "Monday"
holiday = "No"
hour = 14

# 3. Create input DataFrame
new_data = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "SquareFootage": [square_footage],
    "Occupancy": [occupancy],
    "HVACUsage": [hvac_usage],
    "LightingUsage": [lighting_usage],
    "RenewableEnergy": [renewable_energy],
    "DayOfWeek": [day_of_week],
    "Holiday": [holiday],
    "Hour": [hour]
})

# 4. Apply the same encoding as training
new_data = pd.get_dummies(
    new_data,
    drop_first=True,
    dtype=int
)

# 5. Align columns with training features
new_data = new_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# 6. Scale input for Linear Regression
new_data_lr = scaler.transform(new_data)

# 7. Make predictions
lr_prediction = float(lr.predict(new_data_lr)[0])
rf_prediction = float(rf.predict(new_data)[0])

# 8. Calculate electricity cost
lr_cost = lr_prediction * price_per_kwh
rf_cost = rf_prediction * price_per_kwh

# 9. Display results
print("\n======================================")
print("     ENERGY CONSUMPTION PREDICTION")
print("======================================")

print(f"Linear Regression : {lr_prediction:.2f} kWh")
print(f"Random Forest     : {rf_prediction:.2f} kWh")

print("\n======================================")
print("       ESTIMATED ELECTRICITY COST")
print("======================================")

print(f"Price per kWh     : {price_per_kwh} {currency}")
print(f"Linear Regression : {lr_cost:.2f} {currency}")
print(f"Random Forest     : {rf_cost:.2f} {currency}")

print("\nNote: Assumes energy is measured in kWh.")
print("Excludes fixed fees and taxes.")
print("======================================")
