# Problem 4 — Weather Forecast Model (Rainfall Prediction)

## Overview
A machine learning model that predicts rainfall based on weather parameters.
Built using Random Forest Classifier with hyperparameter tuning via GridSearchCV.
Trained on a custom dataset with full ML pipeline — from data collection to model deployment.

## Tech Stack
- Python 3
- Pandas, NumPy
- Scikit-learn (RandomForestClassifier, GridSearchCV, cross_val_score)
- Matplotlib, Seaborn
- Google Colab
- Pickle (model saving and loading)
- Streamlit (web app deployment)

## Dataset
Custom dataset: `Rainfall.csv`

### Raw Features
| Feature       | Description                                      |
| ------------- | ------------------------------------------------ |
| day           | Day identifier (dropped during preprocessing)    |
| pressure      | Atmospheric pressure                             |
| maxtemp       | Maximum temperature (dropped — high correlation) |
| temparature   | Temperature (dropped — high correlation)         |
| mintemp       | Minimum temperature (dropped — high correlation) |
| dewpoint      | Dew point temperature                            |
| humidity      | Humidity percentage                              |
| cloud         | Cloud cover                                      |
| sunshine      | Hours of sunshine                                |
| winddirection | Wind direction                                   |
| windspeed     | Wind speed                                       |
| rainfall      | Target: yes/no → 1/0                             |

### Final Features Used for Training
| Feature       | Description           |
| ------------- | --------------------- |
| pressure      | Atmospheric pressure  |
| dewpoint      | Dew point temperature |
| humidity      | Humidity percentage   |
| cloud         | Cloud cover           |
| sunshine      | Hours of sunshine     |
| winddirection | Wind direction        |
| windspeed     | Wind speed            |

Target: `rainfall` (1 = Rainfall, 0 = No Rainfall)

---

## ML Pipeline

### 1. Data Collection & Loading
- Custom CSV dataset uploaded via Google Colab
- Loaded using Pandas

### 2. Data Preprocessing
- Stripped extra whitespace from all column names
- Dropped `day` column (non-numeric, not useful)
- Checked and handled missing values:
  - `winddirection` → filled with mode
  - `windspeed` → filled with median
- Converted target `rainfall` from yes/no to 1/0
- Dropped highly correlated columns: `maxtemp`, `temparature`, `mintemp`

### 3. Exploratory Data Analysis (EDA)
- Distribution histograms with KDE for all numeric features
- Rainfall class distribution countplot
- Correlation heatmap (coolwarm)
- Boxplots for outlier detection across all features

### 4. Handling Class Imbalance
- Checked class distribution using `value_counts()`
- Separated majority and minority classes
- Downsampled majority class to match minority count using `sklearn.utils.resample`
- Concatenated and shuffled the balanced dataset

### 5. Train/Test Split
- Features: `X` (all columns except rainfall)
- Target: `y` (rainfall)
- Split: 80% train, 20% test (`random_state=42`)

### 6. Model Training
- Algorithm: Random Forest Classifier
- Hyperparameter tuning via GridSearchCV (5-fold cross validation)
- Parameters tuned:

| Parameter         | Values           |
| ----------------- | ---------------- |
| n_estimators      | 50, 100, 200     |
| max_features      | sqrt, log2       |
| max_depth         | None, 10, 20, 30 |
| min_samples_split | 2, 5, 10         |
| min_samples_leaf  | 1, 2, 4          |

- Best parameters selected automatically by GridSearchCV

### 7. Model Evaluation
- 5-fold cross-validation scores on training set
- Mean cross-validation score
- Test set accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

### 8. Prediction on New Data
- Accepts weather input as a tuple
- Returns: `Rainfall` or `No Rainfall`

### 9. Model Saving & Loading
- Model and feature names saved to `rainfall_prediction_model.pkl` using Pickle
- Can be loaded and reused without retraining

---

## Sample Prediction
```python
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
# pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed

# Output: Rainfall
```

---

## How to Run

### On Google Colab (recommended)
1. Open the notebook link below
2. Upload `Rainfall.csv` when prompted
3. Run all cells in order
4. Model saves automatically as `rainfall_prediction_model.pkl`

### Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Or to run the notebook:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```
Then open `Problem_04_Machine_Learning.ipynb` and upload `Rainfall.csv` when prompted.

---

## Files
```
├── Problem_04_Machine_Learning.ipynb   # Main Colab notebook
├── Rainfall.csv                        # Custom dataset
├── rainfall_prediction_model.pkl       # Saved trained model (generated after run)
├── app.py                              # Streamlit web app
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Deployment

### Streamlit App (Live)
[Open Streamlit App](https://ml-problem04-weather-forecast.streamlit.app/)

### Google Colab
[Open in Google Colab](https://colab.research.google.com/drive/1IBPSARZbRqoZG7Nga_QwIIdXfpOZvOOT?usp=sharing)

## GitHub Repository
[Problem 4 — Weather Forecast](https://github.com/agrinova-in/ml-problem04-weather-forecast)