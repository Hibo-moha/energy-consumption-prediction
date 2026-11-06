# Energy Consumption Prediction

## Project Overview

This project uses Machine Learning techniques to predict energy consumption based on environmental conditions, building characteristics, and energy usage patterns.

The project implements two regression algorithms:
- Linear Regression
- Random Forest Regressor

It also includes data preprocessing, feature engineering, model evaluation, prediction, and electricity cost estimation.

## Objectives

- Analyze energy consumption data.
- Clean and preprocess raw datasets.
- Build predictive Machine Learning models.
- Compare model performance using evaluation metrics.
- Estimate electricity costs based on predicted energy consumption.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Dataset

The original dataset contains 1,000 records and 11 columns, including:

- Temperature
- Humidity
- SquareFootage
- Occupancy
- HVACUsage
- LightingUsage
- RenewableEnergy
- DayOfWeek
- Holiday
- Timestamp
- EnergyConsumption (Target Variable)

Feature engineering was applied to extract the hour from timestamps and encode categorical variables.

## Machine Learning Models

Two regression models were trained and evaluated:

1. Linear Regression
2. Random Forest Regressor

## Model Evaluation Results

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 4.10 | 4.42 |
| MSE | 26.61 | 29.74 |
| RMSE | 5.16 | 5.45 |
| R² Score | 0.5938 | 0.5460 |

Linear Regression achieved the best performance among the two tested models.

## Prediction Example

The project predicts energy consumption and estimates electricity costs using an example electricity tariff of $0.50 per kWh.

Example results:

- Linear Regression: 73.94 kWh
- Random Forest: 73.77 kWh
- Estimated cost (Linear Regression): $36.97

These are example predictions based on the provided input values.

## Project Structure

```text
energy-consumption-prediction/
├── dataset/
├── models/
├── outputs/
├── preprocessing.py
├── train.py
├── predict.py
├── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Hibo-moha/energy-consumption-prediction.git
cd energy-consumption-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run data preprocessing

```bash
python preprocessing.py
```

### 4. Train the models

```bash
python train.py
```

### 5. Generate predictions

```bash
python predict.py
```

### 6. Evaluate model performance

```bash
python evaluate.py
```

## Visualization

The evaluation script generates:

- Actual vs Predicted plot
- Residual plot

These visualizations help analyze model performance.

## Future Improvements

- Hyperparameter tuning
- Additional Machine Learning algorithms
- FastAPI model deployment
- Interactive React dashboard
- Improved prediction performance
