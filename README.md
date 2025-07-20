# MLOps Project - Breast Cancer Prediction

This repository contains a complete MLOps pipeline for building, training, and deploying machine learning models for breast cancer diagnosis prediction with experiment tracking, data versioning, and automated deployment capabilities.

## 🚀 Features

- **Data Pipeline**: Automated data ingestion, cleaning, and preprocessing for breast cancer dataset
- **Model Training**: Configurable model training with multiple algorithms (Decision Tree, Random Forest, Gradient Boosting)
- **Experiment Tracking**: MLflow integration for tracking experiments, metrics, and model versioning
- **Data Versioning**: DVC integration for data version control
- **Model Deployment**: FastAPI-based REST API for breast cancer prediction serving
- **Web Interface**: Interactive Streamlit web UI for easy model interaction and visualization
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

1. **Generate breast cancer dataset**:

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
    make api
    # or
    uvicorn app:app --reload
    ```

5. **Launch Streamlit Web UI**:

    ```bash
    make streamlit
    # or 
    streamlit run streamlit_app.py
    ```

6. **Test the API**:

    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{
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
│   ├── preprocessing.py       # Data preprocessing step (cleaning & scaling)
│   ├── train.py               # Model training step
│   └── predict.py             # Prediction step
├── 🤖 models/                  # Trained models
│   └── model.pkl              # Serialized model
├── 📈 mlruns/                  # MLflow experiment tracking
├── 🧪 tests/                   # Test suite
├── 📋 config.yml              # Configuration file
├── 🚀 app.py                  # FastAPI application
├── 🌐 streamlit_app.py        # Streamlit web UI
├── 🐳 dockerfile             # Docker configuration
├── 🔧 main.py                 # Main pipeline script
├── 📊 dataset.py              # Breast cancer data generation script
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

1. **Data Ingestion** (`steps/ingest.py`): Loads breast cancer training and test datasets
2. **Data Preprocessing** (`steps/preprocessing.py`):
   - Feature standardization using StandardScaler
   - Outlier detection and removal (conservative approach for medical data)
   - Data validation and quality checks
3. **Model Training** (`steps/train.py`):
   - SMOTE for handling class imbalance (malignant vs benign)
   - Configurable model training (Decision Tree, Random Forest, Gradient Boosting)
   - Pipeline optimization for breast cancer features
4. **Model Evaluation** (`steps/predict.py`): Generates accuracy, ROC-AUC, precision, and recall metrics for cancer diagnosis

## 📊 Experiment Tracking

The project integrates with MLflow for comprehensive experiment tracking:

- **Metrics**: Accuracy, ROC-AUC, Precision, Recall for cancer diagnosis
- **Parameters**: Model hyperparameters from config
- **Artifacts**: Trained models with input signatures for breast cancer features
- **Model Registry**: Automatic model registration as "breast_cancer_model"

Access the MLflow UI at `http://localhost:5000` after running `make mlflow`.

## 🚀 API Usage

The FastAPI application provides a REST endpoint for breast cancer predictions:

### Endpoints

- **GET** `/`: Health check and API information
- **GET** `/info`: Detailed information about features and model
- **POST** `/predict`: Breast cancer prediction endpoint

### Endpoint: POST `/predict`

**Request Body (all 30 breast cancer features):**

```json
{
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
```

**Response:**

```json
{
  "predicted_class": 0,
  "diagnosis": "Malignant",
  "confidence": {
    "malignant_probability": 0.85,
    "benign_probability": 0.15
  },
  "confidence_score": 0.85
}
```

**Target Classes:**

- `0`: Malignant (cancerous)
- `1`: Benign (non-cancerous)

## 🌐 Streamlit Web UI

The project includes an interactive Streamlit web interface for easy model interaction:

### Features

- **Interactive Input**: User-friendly forms for entering all 30 breast cancer features
- **Multiple Input Methods**:
  - Manual input with sliders and number inputs
  - Load sample data from the breast cancer dataset
  - Generate random samples for testing
  - API testing interface
- **Visual Predictions**: Rich visualizations including probability charts and feature analysis
- **Real-time Validation**: Input validation and error handling
- **Feature Grouping**: Organized feature inputs by categories (Mean, Error, Worst)

### Usage

1. **Start the Streamlit app**:

   ```bash
   make streamlit
   # or
   streamlit run streamlit_app.py
   ```

2. **Access the web interface**: Open your browser and navigate to `http://localhost:8501`

3. **Input Methods**:
   - **Manual Input**: Use the interactive sliders and inputs to enter feature values
   - **Load Sample Data**: Click to load real data samples from the breast cancer dataset
   - **Random Sample**: Generate random feature values within realistic ranges
   - **API Test**: Test the FastAPI endpoint directly from the web interface

4. **View Results**: Get predictions with confidence scores, probability distributions, and feature visualizations

### Screenshots

The Streamlit interface provides:

- Clean, medical-themed UI design
- Organized feature input tabs (Mean Features, Error Features, Worst Features)
- Interactive visualizations with Plotly charts
- Prediction cards with color-coded results (green for benign, red for malignant)
- Feature value overview charts

## 🐳 Docker Deployment

### Single Container

Build and run the application using Docker:

```bash
docker build -t mlops-project .
docker run -p 8000:8000 mlops-project  # FastAPI only
```

### Multi-Service with Docker Compose

Run both API and Streamlit services together:

```bash
docker-compose up --build
```

This will start:

- FastAPI server on `http://localhost:8000`
- Streamlit web UI on `http://localhost:8501`

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
make setup      # Set up virtual environment and install dependencies
make run        # Run the main training pipeline
make mlflow     # Start MLflow UI
make streamlit  # Launch Streamlit web interface
make api        # Start FastAPI server
make test       # Run tests
make clean      # Clean cache and temporary files
make remove     # Remove virtual environment and MLflow runs
```

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE.txt](LICENSE.txt) file for details.
