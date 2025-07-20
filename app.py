from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for predicting breast cancer diagnosis using machine learning",
    version="1.0.0"
)

class BreastCancerInputData(BaseModel):
    """
    Input data model for breast cancer prediction.
    All features are derived from digitized images of breast mass.
    """
    # Mean features
    mean_radius: float = Field(..., ge=0, description="Mean of distances from center to points on the perimeter")
    mean_texture: float = Field(..., ge=0, description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., ge=0, description="Mean perimeter")
    mean_area: float = Field(..., ge=0, description="Mean area")
    mean_smoothness: float = Field(..., ge=0, le=1, description="Mean of local variation in radius lengths")
    mean_compactness: float = Field(..., ge=0, le=1, description="Mean of perimeter^2 / area - 1.0")
    mean_concavity: float = Field(..., ge=0, le=1, description="Mean of severity of concave portions of the contour")
    mean_concave_points: float = Field(..., ge=0, le=1, description="Mean for number of concave portions of the contour")
    mean_symmetry: float = Field(..., ge=0, le=1, description="Mean symmetry")
    mean_fractal_dimension: float = Field(..., ge=0, le=1, description="Mean for coastline approximation - 1")
    
    # Error features (standard error)
    radius_error: float = Field(..., ge=0, description="Standard error for radius")
    texture_error: float = Field(..., ge=0, description="Standard error for texture")
    perimeter_error: float = Field(..., ge=0, description="Standard error for perimeter")
    area_error: float = Field(..., ge=0, description="Standard error for area")
    smoothness_error: float = Field(..., ge=0, description="Standard error for smoothness")
    compactness_error: float = Field(..., ge=0, description="Standard error for compactness")
    concavity_error: float = Field(..., ge=0, description="Standard error for concavity")
    concave_points_error: float = Field(..., ge=0, description="Standard error for concave points")
    symmetry_error: float = Field(..., ge=0, description="Standard error for symmetry")
    fractal_dimension_error: float = Field(..., ge=0, description="Standard error for fractal dimension")
    
    # Worst features (mean of the three largest values)
    worst_radius: float = Field(..., ge=0, description="Worst radius")
    worst_texture: float = Field(..., ge=0, description="Worst texture")
    worst_perimeter: float = Field(..., ge=0, description="Worst perimeter")
    worst_area: float = Field(..., ge=0, description="Worst area")
    worst_smoothness: float = Field(..., ge=0, le=1, description="Worst smoothness")
    worst_compactness: float = Field(..., ge=0, description="Worst compactness")
    worst_concavity: float = Field(..., ge=0, description="Worst concavity")
    worst_concave_points: float = Field(..., ge=0, description="Worst concave points")
    worst_symmetry: float = Field(..., ge=0, description="Worst symmetry")
    worst_fractal_dimension: float = Field(..., ge=0, description="Worst fractal dimension")

    class Config:
        json_schema_extra = {
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184,
                "mean_compactness": 0.2776,
                "mean_concavity": 0.3001,
                "mean_concave_points": 0.1471,
                "mean_symmetry": 0.2419,
                "mean_fractal_dimension": 0.07871,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189
            }
        }

try:
    model = joblib.load('models/model.pkl')
except FileNotFoundError:
    model = None
    print("Warning: Model file not found. Please train and save a model first.")

@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {
        "health_check": "OK", 
        "model_version": "1.0.0",
        "description": "Breast Cancer Prediction API",
        "model_loaded": model is not None
    }

@app.get("/info")
async def get_model_info():
    """Get information about the model and expected input features"""
    feature_info = {
        "total_features": 30,
        "feature_groups": {
            "mean_features": [
                "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
                "mean_smoothness", "mean_compactness", "mean_concavity", 
                "mean_concave_points", "mean_symmetry", "mean_fractal_dimension"
            ],
            "error_features": [
                "radius_error", "texture_error", "perimeter_error", "area_error",
                "smoothness_error", "compactness_error", "concavity_error",
                "concave_points_error", "symmetry_error", "fractal_dimension_error"
            ],
            "worst_features": [
                "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
                "worst_smoothness", "worst_compactness", "worst_concavity",
                "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
            ]
        },
        "target_classes": {
            "0": "Malignant",
            "1": "Benign"
        }
    }
    return feature_info

@app.post("/predict")
async def predict(input_data: BreastCancerInputData):
    """
    Predict breast cancer diagnosis based on input features.
    Returns 0 for malignant, 1 for benign.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train and save a model first.")
    
    try:
        # Convert input data to DataFrame with correct column names
        input_dict = input_data.model_dump()
        
        # Map pydantic field names to actual dataset column names
        column_mapping = {
            'mean_radius': 'mean radius',
            'mean_texture': 'mean texture',
            'mean_perimeter': 'mean perimeter',
            'mean_area': 'mean area',
            'mean_smoothness': 'mean smoothness',
            'mean_compactness': 'mean compactness',
            'mean_concavity': 'mean concavity',
            'mean_concave_points': 'mean concave points',
            'mean_symmetry': 'mean symmetry',
            'mean_fractal_dimension': 'mean fractal dimension',
            'radius_error': 'radius error',
            'texture_error': 'texture error',
            'perimeter_error': 'perimeter error',
            'area_error': 'area error',
            'smoothness_error': 'smoothness error',
            'compactness_error': 'compactness error',
            'concavity_error': 'concavity error',
            'concave_points_error': 'concave points error',
            'symmetry_error': 'symmetry error',
            'fractal_dimension_error': 'fractal dimension error',
            'worst_radius': 'worst radius',
            'worst_texture': 'worst texture',
            'worst_perimeter': 'worst perimeter',
            'worst_area': 'worst area',
            'worst_smoothness': 'worst smoothness',
            'worst_compactness': 'worst compactness',
            'worst_concavity': 'worst concavity',
            'worst_concave_points': 'worst concave points',
            'worst_symmetry': 'worst symmetry',
            'worst_fractal_dimension': 'worst fractal dimension'
        }
        
        # Create DataFrame with proper column names
        mapped_data = {column_mapping[key]: value for key, value in input_dict.items()}
        df = pd.DataFrame([mapped_data])
        
        # Make prediction
        pred = model.predict(df)
        pred_proba = model.predict_proba(df)
        
        # Get prediction probabilities
        malignant_prob = float(pred_proba[0][0])
        benign_prob = float(pred_proba[0][1])
        
        return {
            "predicted_class": int(pred[0]),
            "diagnosis": "Malignant" if pred[0] == 0 else "Benign",
            "confidence": {
                "malignant_probability": malignant_prob,
                "benign_probability": benign_prob
            },
            "confidence_score": max(malignant_prob, benign_prob)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


