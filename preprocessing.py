
import os
import pandas as pd

CSV_PATH = "./dataset/Energy_consumption.csv"
OUT_PATH = "./dataset/clean_energy.csv"

# 1. Load dataset
df = pd.read_csv(CSV_PATH)

print("\n=== INITIAL DATA ===")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Remove duplicates
df = df.drop_duplicates().copy()

# 3. Convert Timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(subset=["Timestamp"])

# 4. Feature engineering
df["Hour"] = df["Timestamp"].dt.hour

# 5. Encode binary categorical columns
binary_columns = {
    "HVACUsage": {"Off": 0, "On": 1},
    "LightingUsage": {"Off": 0, "On": 1},
    "Holiday": {"No": 0, "Yes": 1}
}

for col, mapping in binary_columns.items():
    df[col] = df[col].map(mapping)

# 6. One-hot encode DayOfWeek
df = pd.get_dummies(
    df,
    columns=["DayOfWeek"],
    dtype=int
)

# 7. Drop Timestamp
df = df.drop(columns=["Timestamp"])

# 8. Handle missing values
for col in df.select_dtypes(include=["number"]).columns:
    df[col] = df[col].fillna(df[col].median())

# 9. Validate target
if "EnergyConsumption" not in df.columns:
    raise ValueError("EnergyConsumption column is missing!")

if df["EnergyConsumption"].isnull().any():
    raise ValueError("Target contains missing values!")

# 10. Final dataset information
print("\n=== CLEAN DATA ===")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# 11. Save cleaned dataset
os.makedirs("./dataset", exist_ok=True)

df.to_csv(OUT_PATH, index=False)

print("\nPreprocessing completed successfully!")
print(f"Saved dataset to: {OUT_PATH}")
