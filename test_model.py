import joblib
import pandas as pd
import numpy as np

model = joblib.load('models/west_africa_model.joblib')
feature_cols = joblib.load('models/west_africa_feature_cols.joblib')

sample_input = {
    'tmin_temperature': 20.0, 
    'tmax_temperature': 35.0, 
    'min_precipitation': 50.0, 
    'max_aod': 1350.0, 
    'precip_range': 125.0, 
    'max_aod_lag1': 1370.0, 
    'max_aod_lag3': 1369.0, 
    'min_precipitation_lag1': 51.0, 
    'max_aod_roll6': 1367.0, 
    'min_precipitation_roll6': 51.0, 
    'month_sin': 0.5, 
    'month_cos': 0.87
}

input_df = pd.DataFrame([sample_input], columns=feature_cols)
prediction = model.predict(input_df)
raw_value = prediction[0]
cleaned_value = np.round(np.clip(raw_value, 0, None), 0)
print(cleaned_value)