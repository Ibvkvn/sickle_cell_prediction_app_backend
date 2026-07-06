from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

REGION_MODELS = {
    "west africa": "models/west_africa_model.joblib",
    "north africa": "models/north_africa_model.joblib",
    "east africa": "models/east_africa_model.joblib",
    "south africa": "models/southern_africa_model.joblib",
    "central africa": "models/central_africa_model.joblib",    
}

loaded_models = {}
loaded_features = {}

for region, path in REGION_MODELS.items():
    loaded_models[region] = joblib.load(path)
    loaded_features[region] = joblib.load(path.replace("_model.joblib", "_feature_cols.joblib"))

class PredictRequest(BaseModel):
    region: str
    month: int
    features: dict

@app.post("/predict")
def predict(request: PredictRequest):
    if request.region not in loaded_models:
        raise HTTPException(status_code= 400, detail="invalid region")

    model= loaded_models[request.region]
    feature_cols= loaded_features[request.region]

    missing = [col for col in feature_cols if col not in request.features]
    if missing: 
        raise HTTPException(status_code= 400, detail=f"missing features {missing}")

    input_df = pd.DataFrame([request.features], columns=feature_cols)
    model_result = model.predict(input_df)
    final_result = np.round((np.clip(model_result, 0, None)), 0)
    predicted_value = float(final_result[0])
    return {"predicted_crisis_rate": predicted_value}

@app.get("/health")
def get_health():
    return {"status": "ok"}

