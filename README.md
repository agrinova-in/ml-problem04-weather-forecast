# Problem 4 — Weather Forecast Model (Rainfall Prediction)

## Overview
A machine learning model that predicts rainfall based on weather parameters.
Built using Random Forest Classifier with hyperparameter tuning via GridSearchCV.

> Note: The problem title says ARIMA but the implementation uses Random Forest
> Classification on a rainfall dataset, which better suits the tabular weather data.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (RandomForestClassifier, GridSearchCV)
- Matplotlib, Seaborn
- Google Colab
- Pickle (model saving)

## Dataset
Custom dataset: `Rainfall.csv`
Features used after preprocessing:
| Feature | Description |
|---|---|
| pressure | Atmospheric pressure |
| dewpoint | Dew point temperature |
| humidity | Humidity percentage |
| cloud | Cloud cover |
| sunshine | Hours of sunshine |
| winddirection | Wind direction |
| windspeed | Wind speed |

Target: `rainfall` (1 = Yes, 0 = No)

## ML Pipeline

### 1. Data Collection & Preprocessing
- Loaded custom CSV dataset
- Removed extra whitespace from column names
- Dropped `day` column (non-numeric)
- Handled missing values:
  - `winddirection` → filled with mode
  - `windspeed` → filled with median
- Converted `rainfall` from yes/no to 1/0
- Dropped highly correlated columns: `maxtemp`, `temparature`, `mintemp`

### 2. Exploratory Data Analysis
- Distribution plots for all numeric features
- Rainfall class distribution (countplot)
- Correlation heatmap
- Boxplots for outlier detection

### 3. Handling Class Imbalance
- Identified majority/minority class
- Downsampled majority class to match minority count
- Shuffled final balanced dataset

### 4. Model Training
- Algorithm: Random Forest Classifier
- Hyperparameter tuning: GridSearchCV with 5-fold cross validation
- Parameters tuned:
  - `n_estimators`: [50, 100, 200]
  - `max_features`: [sqrt, log2]
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

### 5. Model Evaluation
- Cross-validation scores (5-fold)
- Test set accuracy
- Confusion matrix
- Classification report (precision, recall, F1)

### 6. Prediction
- Accepts new weather input as tuple
- Returns: `Rainfall` or `No Rainfall`

### 7. Model Saving
- Saved model + feature names to `rainfall_prediction_model.pkl` using Pickle
- Can be loaded and reused without retraining

## Sample Prediction
```python
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
# Output: Rainfall
```

## How to Run

### On Google Colab (recommended)
1. Open the notebook in Google Colab
2. Upload `Rainfall.csv` when prompted
3. Run all cells in order
4. Model saves as `rainfall_prediction_model.pkl`

### Locally
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook
```
Upload `Rainfall.csv` and run all cells.

## Files
```
├── weather_forecast.ipynb     # Main notebook
├── Rainfall.csv               # Dataset
├── rainfall_prediction_model.pkl  # Saved model (generated after run)
└── README.md
```

## Deployment Link
[Google Colab Notebook](#) ← paste your Colab share link here

## GitHub Repository
[Problem 4 — Weather Forecast](https://github.com/agrinova-in/Weather-Forecast-ARIMA)