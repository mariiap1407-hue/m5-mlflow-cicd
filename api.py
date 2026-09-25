from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import os


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
model_uri = os.getenv("MODEL_URI", "models:/diabetes-risk-model@production")
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI()

class PatientProfile(BaseModel):
    pregnancies: float = Field(ge=0, le=20)
    glucose: float = Field(gt=0, le=400)
    blood_pressure: float = Field(gt=0, le=200)
    skin_thickness: float = Field(ge=0, le=100)
    insulin: float = Field(ge=0, le=900)
    bmi: float = Field(gt=0, le=80)
    diabetes_pedigree: float = Field(ge=0, le=3)
    age: float = Field(ge=0, le=120)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PatientProfile):
    data = pd.DataFrame([payload.model_dump()])
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    return {"prediction": prediction, "probability": probability}

