from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI()

class InputData(BaseModel):
    Gender: str = Field(..., pattern="^(Male|Female)$")
    Age: int = Field(..., ge=18, le=100)  # Age between 18-100
    HasDrivingLicense: int = Field(..., ge=0, le=1)  # Only 0 or 1
    RegionID: float = Field(..., gt=0)
    Switch: int = Field(..., ge=0, le=1)
    PastAccident: str = Field(..., pattern="^(Yes|No|Unknown)$")
    AnnualPremium: float = Field(..., gt=0) 

model = joblib.load('models/model.pkl')

@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}

@app.post("/predict")
async def predict(input_data: InputData):
    
        df = pd.DataFrame([input_data.model_dump().values()], 
                          columns=list(input_data.model_dump().keys()))
        pred = model.predict(df)
        return {"predicted_class": int(pred[0])}


