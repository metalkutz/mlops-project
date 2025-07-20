import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from steps.ingest import Ingestion

# Sample configuration data for breast cancer dataset
@pytest.fixture
def config_data():
    return {
        'data': {
            'train_path': 'data/train.csv',
            'test_path': 'data/test.csv'
        },
        'train': {
            'test_size': 0.2,
            'random_state': 42,
            'shuffle': True
        },
        'model': {
            'name': 'DecisionTreeClassifier',
            'params': {
                'criterion': 'entropy',
                'max_depth': None
            },
            'store_path': 'models/'
        }
    }

# Sample breast cancer dataset structure
@pytest.fixture
def sample_data():
    """
    Create sample data that mimics the breast cancer dataset structure
    """
    # Sample train data with breast cancer features
    train_data = pd.DataFrame({
        'mean radius': [14.0, 20.0],
        'mean texture': [19.0, 17.8],
        'mean perimeter': [91.0, 132.9],
        'mean area': [600.0, 1326.0],
        'mean smoothness': [0.11, 0.085],
        'mean compactness': [0.15, 0.079],
        'mean concavity': [0.08, 0.087],
        'mean concave points': [0.04, 0.07],
        'mean symmetry': [0.18, 0.181],
        'mean fractal dimension': [0.063, 0.057],
        'radius error': [0.4, 1.1],
        'texture error': [1.2, 0.9],
        'perimeter error': [2.5, 8.6],
        'area error': [25.0, 153.4],
        'smoothness error': [0.007, 0.0064],
        'compactness error': [0.02, 0.049],
        'concavity error': [0.025, 0.054],
        'concave points error': [0.01, 0.016],
        'symmetry error': [0.02, 0.03],
        'fractal dimension error': [0.004, 0.0062],
        'worst radius': [16.0, 25.4],
        'worst texture': [25.0, 17.3],
        'worst perimeter': [105.0, 184.6],
        'worst area': [850.0, 2019.0],
        'worst smoothness': [0.14, 0.162],
        'worst compactness': [0.25, 0.666],
        'worst concavity': [0.2, 0.712],
        'worst concave points': [0.12, 0.265],
        'worst symmetry': [0.28, 0.46],
        'worst fractal dimension': [0.09, 0.119],
        'target': [0, 0]  # 0 = malignant, 1 = benign
    })
    
    # Sample test data with breast cancer features
    test_data = pd.DataFrame({
        'mean radius': [12.0, 18.0],
        'mean texture': [23.0, 20.4],
        'mean perimeter': [78.0, 122.8],
        'mean area': [450.0, 1001.0],
        'mean smoothness': [0.095, 0.118],
        'mean compactness': [0.12, 0.278],
        'mean concavity': [0.06, 0.30],
        'mean concave points': [0.03, 0.147],
        'mean symmetry': [0.165, 0.242],
        'mean fractal dimension': [0.068, 0.079],
        'radius error': [0.3, 1.095],
        'texture error': [1.5, 0.905],
        'perimeter error': [2.0, 8.589],
        'area error': [20.0, 153.4],
        'smoothness error': [0.008, 0.0064],
        'compactness error': [0.015, 0.049],
        'concavity error': [0.02, 0.054],
        'concave points error': [0.008, 0.016],
        'symmetry error': [0.018, 0.03],
        'fractal dimension error': [0.003, 0.0062],
        'worst radius': [14.0, 25.38],
        'worst texture': [28.0, 17.33],
        'worst perimeter': [90.0, 184.6],
        'worst area': [650.0, 2019.0],
        'worst smoothness': [0.13, 0.162],
        'worst compactness': [0.2, 0.666],
        'worst concavity': [0.15, 0.712],
        'worst concave points': [0.08, 0.265],
        'worst symmetry': [0.25, 0.46],
        'worst fractal dimension': [0.08, 0.119],
        'target': [1, 0]  # 0 = malignant, 1 = benign
    })
    
    return train_data, test_data

@patch("builtins.open", new_callable=mock_open, read_data="dummy")
@patch("yaml.safe_load")
@patch("pandas.read_csv")
def test_load_data(mock_read_csv, mock_safe_load, mock_open, config_data, sample_data):
    """Test data loading functionality with breast cancer dataset structure"""
    # Mock the YAML safe_load to return the sample config data
    mock_safe_load.return_value = config_data

    # Mock the read_csv to return the sample dataframes
    mock_read_csv.side_effect = sample_data

    ingestion = Ingestion()
    train_data, test_data = ingestion.load_data()

    # Check if the dataframes returned are as expected
    pd.testing.assert_frame_equal(train_data, sample_data[0])
    pd.testing.assert_frame_equal(test_data, sample_data[1])

    # Verify the correct file paths were read (matching config.yml)
    mock_read_csv.assert_any_call('data/train.csv')
    mock_read_csv.assert_any_call('data/test.csv')
    
    # Verify the data structure matches breast cancer dataset
    assert 'target' in train_data.columns
    assert 'target' in test_data.columns
    assert train_data.shape[1] == 31  # 30 features + 1 target
    assert test_data.shape[1] == 31   # 30 features + 1 target
    
    # Verify target values are binary (0 or 1)
    assert set(train_data['target'].unique()).issubset({0, 1})
    assert set(test_data['target'].unique()).issubset({0, 1})

def test_breast_cancer_data_structure(sample_data):
    """Test that the sample data has the correct breast cancer dataset structure"""
    train_data, test_data = sample_data
    
    # Check that all expected breast cancer features are present
    expected_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 
        'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 
        'area error', 'smoothness error', 'compactness error', 'concavity error', 
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 
        'worst smoothness', 'worst compactness', 'worst concavity', 
        'worst concave points', 'worst symmetry', 'worst fractal dimension', 'target'
    ]
    
    # Verify all expected columns are present
    for feature in expected_features:
        assert feature in train_data.columns, f"Feature '{feature}' missing from train data"
        assert feature in test_data.columns, f"Feature '{feature}' missing from test data"
    
    # Verify data types are numeric
    for col in train_data.columns:
        assert pd.api.types.is_numeric_dtype(train_data[col]), f"Column '{col}' should be numeric"
        assert pd.api.types.is_numeric_dtype(test_data[col]), f"Column '{col}' should be numeric"
    
    # Verify no missing values
    assert not train_data.isnull().any().any(), "Train data should not contain missing values"
    assert not test_data.isnull().any().any(), "Test data should not contain missing values"

@patch("builtins.open", new_callable=mock_open, read_data="dummy")
@patch("yaml.safe_load")
def test_config_loading(mock_safe_load, mock_open, config_data):
    """Test configuration loading functionality"""
    mock_safe_load.return_value = config_data
    
    ingestion = Ingestion()
    
    # Verify config is loaded correctly
    assert ingestion.config['data']['train_path'] == 'data/train.csv'
    assert ingestion.config['data']['test_path'] == 'data/test.csv'