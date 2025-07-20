# MLOps Project

This repository contains a complete MLOps pipeline for building, training, and deploying machine learning models with experiment tracking, data versioning, and automated deployment capabilities.

## 🚀 Features

- **Data Pipeline**: Automated data ingestion, cleaning, and preprocessing
- **Model Training**: Configurable model training with multiple algorithms (Decision Tree, Random Forest, Gradient Boosting)
- **Experiment Tracking**: MLflow integration for tracking experiments, metrics, and model versioning
- **Data Versioning**: DVC integration for data version control
- **Model Deployment**: FastAPI-based REST API for model serving
- **Testing**: Comprehensive test suite with pytest
- **Containerization**: Docker support for deployment
- **Pipeline Orchestration**: Modular pipeline architecture

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- Git
- Docker (optional)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/metalkutz/mlops-project.git
    cd mlops-project
    ```

2. Set up virtual environment and install dependencies:

    ```bash
    make setup
    ```
    
    Or manually:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

### Quick Start

1. **Generate sample data** (optional):

    ```bash
    python dataset.py
    ```

2. **Run the training pipeline**:

    ```bash
    make run
    # or
    python main.py
    ```

3. **Start MLflow UI** to view experiments:

    ```bash
    make mlflow
    # or
    mlflow ui
    ```

4. **Deploy the model API**:

    ```bash
    uvicorn app:app --reload
    ```

5. **Test the API**:

    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{
           "Gender": "Male",
           "Age": 30,
           "HasDrivingLicense": 1,
           "RegionID": 1.0,
           "Switch": 0,
           "PastAccident": "No",
           "AnnualPremium": 25000.0
         }'
    ```

## 📁 Project Structure

```text
mlops-project/
├── 📊 data/                    # Data directory
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   └── production.csv         # Production data
├── 🔧 steps/                   # Pipeline components
│   ├── __init__.py
│   ├── ingest.py              # Data ingestion step
│   ├── clean.py               # Data cleaning step
│   ├── train.py               # Model training step
│   └── predict.py             # Prediction step
├── 🤖 models/                  # Trained models
│   └── model.pkl              # Serialized model
├── 📈 mlruns/                  # MLflow experiment tracking
├── 🧪 tests/                   # Test suite
├── 📋 config.yml              # Configuration file
├── 🚀 app.py                  # FastAPI application
├── 🐳 dockerfile             # Docker configuration
├── 🔧 main.py                 # Main pipeline script
├── 📊 dataset.py              # Data generation script
├── 🏗️ Makefile               # Build automation
├── 📦 requirements.txt        # Python dependencies
├── 📋 data.dvc               # DVC data tracking
├── 📄 LICENSE.txt            # Apache 2.0 License
└── 📖 README.md              # This file
```

## ⚙️ Configuration

The project uses `config.yml` for configuration. You can modify model parameters, data paths, and training settings:

```yaml
model:
  name: DecisionTreeClassifier  # or RandomForestClassifier, GradientBoostingClassifier
  params:
    criterion: entropy
    max_depth: null
  store_path: models/
```

## 🧠 Model Pipeline

The ML pipeline consists of several modular steps:

1. **Data Ingestion** (`steps/ingest.py`): Loads training and test datasets
2. **Data Cleaning** (`steps/clean.py`): Handles missing values and data preprocessing
3. **Model Training** (`steps/train.py`):
   - Preprocessing with StandardScaler, OneHotEncoder, MinMaxScaler
   - SMOTE for handling class imbalance
   - Configurable model training (Decision Tree, Random Forest, Gradient Boosting)
4. **Model Evaluation** (`steps/predict.py`): Generates accuracy, ROC-AUC, precision, and recall metrics

## 📊 Experiment Tracking

The project integrates with MLflow for comprehensive experiment tracking:

- **Metrics**: Accuracy, ROC-AUC, Precision, Recall
- **Parameters**: Model hyperparameters from config
- **Artifacts**: Trained models with input signatures
- **Model Registry**: Automatic model registration as "insurance_model"

Access the MLflow UI at `http://localhost:5000` after running `make mlflow`.

## 🚀 API Usage

The FastAPI application provides a REST endpoint for predictions:

### Endpoint: POST `/predict`

**Request Body:**

```json
{
  "Gender": "Male",
  "Age": 30,
  "HasDrivingLicense": 1,
  "RegionID": 1.0,
  "Switch": 0,
  "PastAccident": "No",
  "AnnualPremium": 25000.0
}
```

**Response:**

```json
{
  "predicted_class": 1
}
```

## 🐳 Docker Deployment

Build and run the application using Docker:

```bash
docker build -t mlops-project .
docker run -p 8000:80 mlops-project
```

## 🧪 Testing

Run the test suite:

```bash
make test
# or
pytest
```

## 📦 Data Version Control

The project uses DVC for data versioning. The `data.dvc` file tracks data changes:

```bash
dvc init  # initialize a DVC
dvc add data # track data files

#add remote storage configuration
dvc remote add -d <remote_name> <remote_storage_path>

dvc pull  # Download data
dvc push  # Upload data changes
```

## 🛠️ Development Commands

Available Makefile commands:

```bash
make setup    # Set up virtual environment and install dependencies
make run      # Run the main training pipeline
make mlflow   # Start MLflow UI
make test     # Run tests
make clean    # Clean cache and temporary files
make remove   # Remove virtual environment and MLflow runs
```

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE.txt](LICENSE.txt) file for details.
