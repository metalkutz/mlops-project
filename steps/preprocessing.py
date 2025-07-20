import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocess_data:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data):
        """
        Clean the breast cancer dataset.
        The dataset is already clean with no missing values,
        but we apply feature scaling for better model performance.
        """
        # Make a copy to avoid modifying the original data
        cleaned_data = data.copy()
        
        # Separate features and target
        feature_columns = [col for col in cleaned_data.columns if col != 'target']
        
        # Check for any outliers using IQR method and optionally remove extreme outliers
        # For medical data, we're more conservative about removing outliers
        for col in feature_columns:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Use a more conservative multiplier (3.0 instead of 1.5) for medical data
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR
            
            # Only remove extreme outliers that are likely data errors
            cleaned_data = cleaned_data[
                (cleaned_data[col] >= lower_bound) & 
                (cleaned_data[col] <= upper_bound)
            ]
        
        # Apply feature scaling to all numeric features
        # This is important for algorithms sensitive to feature scales
        cleaned_data[feature_columns] = self.scaler.fit_transform(cleaned_data[feature_columns])
        
        print(f"Data cleaning completed. Shape: {cleaned_data.shape}")
        print(f"Target distribution: {cleaned_data['target'].value_counts().to_dict()}")
        
        return cleaned_data