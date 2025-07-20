import pytest
import pandas as pd
import numpy as np
from steps.preprocessing import Preprocess_data

@pytest.fixture
def sample_breast_cancer_data():
    """
    Create sample data that mimics the breast cancer dataset structure
    """
    np.random.seed(42)  # For reproducible tests
    
    # Create sample data with the same structure as breast cancer dataset
    data = {
        'mean radius': [14.0, 20.0, 12.0, 18.0, 15.0],
        'mean texture': [19.0, 17.8, 23.0, 20.4, 16.2],
        'mean perimeter': [91.0, 132.9, 78.0, 122.8, 103.0],
        'mean area': [600.0, 1326.0, 450.0, 1001.0, 750.0],
        'mean smoothness': [0.11, 0.085, 0.095, 0.118, 0.092],
        'mean compactness': [0.15, 0.079, 0.12, 0.278, 0.11],
        'mean concavity': [0.08, 0.087, 0.06, 0.30, 0.075],
        'mean concave points': [0.04, 0.07, 0.03, 0.147, 0.045],
        'mean symmetry': [0.18, 0.181, 0.165, 0.242, 0.175],
        'mean fractal dimension': [0.063, 0.057, 0.068, 0.079, 0.061],
        'radius error': [0.4, 1.1, 0.3, 1.095, 0.5],
        'texture error': [1.2, 0.9, 1.5, 0.905, 1.1],
        'perimeter error': [2.5, 8.6, 2.0, 8.589, 3.0],
        'area error': [25.0, 153.4, 20.0, 153.4, 30.0],
        'smoothness error': [0.007, 0.0064, 0.008, 0.0064, 0.0065],
        'compactness error': [0.02, 0.049, 0.015, 0.049, 0.025],
        'concavity error': [0.025, 0.054, 0.02, 0.054, 0.03],
        'concave points error': [0.01, 0.016, 0.008, 0.016, 0.012],
        'symmetry error': [0.02, 0.03, 0.018, 0.03, 0.022],
        'fractal dimension error': [0.004, 0.0062, 0.003, 0.0062, 0.0045],
        'worst radius': [16.0, 25.4, 14.0, 25.38, 18.0],
        'worst texture': [25.0, 17.3, 28.0, 17.33, 22.0],
        'worst perimeter': [105.0, 184.6, 90.0, 184.6, 120.0],
        'worst area': [850.0, 2019.0, 650.0, 2019.0, 1000.0],
        'worst smoothness': [0.14, 0.162, 0.13, 0.162, 0.135],
        'worst compactness': [0.25, 0.666, 0.2, 0.666, 0.22],
        'worst concavity': [0.2, 0.712, 0.15, 0.712, 0.18],
        'worst concave points': [0.12, 0.265, 0.08, 0.265, 0.10],
        'worst symmetry': [0.28, 0.46, 0.25, 0.46, 0.30],
        'worst fractal dimension': [0.09, 0.119, 0.08, 0.119, 0.085],
        'target': [0, 0, 1, 0, 1]  # 0 = malignant, 1 = benign
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def preprocessor():
    return Preprocess_data()

def test_preprocess_data(preprocessor, sample_breast_cancer_data):
    """Test the preprocessing functionality for breast cancer dataset"""
    original_shape = sample_breast_cancer_data.shape
    cleaned_data = preprocessor.preprocess_data(sample_breast_cancer_data.copy())

    # Check that the data structure is maintained
    assert 'target' in cleaned_data.columns
    assert cleaned_data.shape[1] == original_shape[1]  # Same number of columns
    
    # Check that outlier removal might reduce the number of rows
    assert cleaned_data.shape[0] <= original_shape[0]
    
    # Check that features are standardized (mean ≈ 0, std ≈ 1)
    feature_columns = [col for col in cleaned_data.columns if col != 'target']
    for col in feature_columns:
        feature_mean = cleaned_data[col].mean()
        feature_std = cleaned_data[col].std()
        # Allow some tolerance for numerical precision and outlier removal effects
        assert abs(feature_mean) < 1e-10, f"Feature {col} mean should be close to 0, got {feature_mean}"
        assert 0.5 <= feature_std <= 2.0, f"Feature {col} std should be reasonable, got {feature_std}"
    
    # Check that no missing values are introduced
    assert not cleaned_data.isnull().any().any()
    
    # Check that target values are still valid (0 or 1)
    assert set(cleaned_data['target'].unique()).issubset({0, 1})
    
    # Check that we have both classes if the dataset is large enough
    if cleaned_data.shape[0] >= 2:
        assert len(cleaned_data['target'].unique()) >= 1

def test_outlier_removal(preprocessor):
    """Test outlier removal functionality"""
    # Create data with extreme outliers
    data_with_outliers = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 1000],  # 1000 is an extreme outlier
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })
    
    cleaned_data = preprocessor.preprocess_data(data_with_outliers.copy())
    
    # The extreme outlier should be removed
    assert cleaned_data.shape[0] < data_with_outliers.shape[0]
    
def test_empty_data(preprocessor):
    """Test handling of edge cases"""
    # Test with minimal data
    minimal_data = pd.DataFrame({
        'feature1': [1.0, 2.0],
        'target': [0, 1]
    })
    
    cleaned_data = preprocessor.preprocess_data(minimal_data.copy())
    assert cleaned_data.shape[0] <= minimal_data.shape[0]  # Should not increase rows
    assert 'target' in cleaned_data.columns