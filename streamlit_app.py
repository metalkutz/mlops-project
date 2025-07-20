import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

# Page config
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .benign {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .malignant {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .feature-group {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Breast Cancer Diagnosis Predictor</h1>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please train and save a model first by running `python main.py`")
        return None

# Feature definitions
FEATURE_GROUPS = {
    "Mean Features": {
        "mean_radius": {"min": 6.0, "max": 30.0, "default": 14.13, "step": 0.1},
        "mean_texture": {"min": 9.0, "max": 40.0, "default": 19.29, "step": 0.1},
        "mean_perimeter": {"min": 40.0, "max": 200.0, "default": 91.97, "step": 0.1},
        "mean_area": {"min": 140.0, "max": 2500.0, "default": 654.89, "step": 1.0},
        "mean_smoothness": {"min": 0.05, "max": 0.17, "default": 0.096, "step": 0.001},
        "mean_compactness": {"min": 0.02, "max": 0.35, "default": 0.104, "step": 0.001},
        "mean_concavity": {"min": 0.0, "max": 0.43, "default": 0.089, "step": 0.001},
        "mean_concave_points": {"min": 0.0, "max": 0.20, "default": 0.048, "step": 0.001},
        "mean_symmetry": {"min": 0.11, "max": 0.30, "default": 0.181, "step": 0.001},
        "mean_fractal_dimension": {"min": 0.05, "max": 0.10, "default": 0.063, "step": 0.001}
    },
    "Error Features": {
        "radius_error": {"min": 0.1, "max": 3.0, "default": 0.405, "step": 0.01},
        "texture_error": {"min": 0.4, "max": 5.0, "default": 1.217, "step": 0.01},
        "perimeter_error": {"min": 0.8, "max": 22.0, "default": 2.866, "step": 0.01},
        "area_error": {"min": 6.0, "max": 550.0, "default": 40.34, "step": 0.1},
        "smoothness_error": {"min": 0.002, "max": 0.032, "default": 0.007, "step": 0.0001},
        "compactness_error": {"min": 0.002, "max": 0.14, "default": 0.025, "step": 0.001},
        "concavity_error": {"min": 0.0, "max": 0.40, "default": 0.032, "step": 0.001},
        "concave_points_error": {"min": 0.0, "max": 0.053, "default": 0.012, "step": 0.0001},
        "symmetry_error": {"min": 0.008, "max": 0.08, "default": 0.021, "step": 0.001},
        "fractal_dimension_error": {"min": 0.0008, "max": 0.030, "default": 0.004, "step": 0.0001}
    },
    "Worst Features": {
        "worst_radius": {"min": 7.0, "max": 37.0, "default": 16.27, "step": 0.1},
        "worst_texture": {"min": 12.0, "max": 50.0, "default": 25.68, "step": 0.1},
        "worst_perimeter": {"min": 50.0, "max": 252.0, "default": 107.26, "step": 0.1},
        "worst_area": {"min": 185.0, "max": 4255.0, "default": 880.58, "step": 1.0},
        "worst_smoothness": {"min": 0.071, "max": 0.223, "default": 0.132, "step": 0.001},
        "worst_compactness": {"min": 0.027, "max": 1.06, "default": 0.254, "step": 0.001},
        "worst_concavity": {"min": 0.0, "max": 1.25, "default": 0.272, "step": 0.001},
        "worst_concave_points": {"min": 0.0, "max": 0.29, "default": 0.115, "step": 0.001},
        "worst_symmetry": {"min": 0.156, "max": 0.664, "default": 0.290, "step": 0.001},
        "worst_fractal_dimension": {"min": 0.055, "max": 0.208, "default": 0.084, "step": 0.001}
    }
}

# Sidebar for input method selection
st.sidebar.title("🎛️ Input Method")
input_method = st.sidebar.radio(
    "Choose how to input data:",
    ["Manual Input", "Load Sample Data", "Random Sample", "API Test"]
)

# Load model
model = load_model()

if model is None:
    st.stop()

# Function to create feature inputs
def create_feature_inputs():
    feature_values = {}
    
    # Create tabs for feature groups
    tabs = st.tabs(list(FEATURE_GROUPS.keys()))
    
    for i, (group_name, features) in enumerate(FEATURE_GROUPS.items()):
        with tabs[i]:
            st.markdown(f'<div class="feature-group">', unsafe_allow_html=True)
            st.subheader(f"📊 {group_name}")
            
            # Create columns for better layout
            cols = st.columns(2)
            
            for j, (feature_name, config) in enumerate(features.items()):
                with cols[j % 2]:
                    # Format feature name for display
                    display_name = feature_name.replace('_', ' ').title()
                    feature_values[feature_name] = st.number_input(
                        f"{display_name}",
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"],
                        step=config["step"],
                        key=f"input_{feature_name}"
                    )
            st.markdown('</div>', unsafe_allow_html=True)
    
    return feature_values

# Function to load sample data
def load_sample_data():
    """Load sample data from breast cancer dataset"""
    from sklearn.datasets import load_breast_cancer
    
    # Load the dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Get a random sample
    sample_idx = np.random.randint(0, len(df))
    sample = df.iloc[sample_idx]
    
    # Convert to our feature format
    feature_mapping = {
        'mean radius': 'mean_radius',
        'mean texture': 'mean_texture',
        'mean perimeter': 'mean_perimeter',
        'mean area': 'mean_area',
        'mean smoothness': 'mean_smoothness',
        'mean compactness': 'mean_compactness',
        'mean concavity': 'mean_concavity',
        'mean concave points': 'mean_concave_points',
        'mean symmetry': 'mean_symmetry',
        'mean fractal dimension': 'mean_fractal_dimension',
        'radius error': 'radius_error',
        'texture error': 'texture_error',
        'perimeter error': 'perimeter_error',
        'area error': 'area_error',
        'smoothness error': 'smoothness_error',
        'compactness error': 'compactness_error',
        'concavity error': 'concavity_error',
        'concave points error': 'concave_points_error',
        'symmetry error': 'symmetry_error',
        'fractal dimension error': 'fractal_dimension_error',
        'worst radius': 'worst_radius',
        'worst texture': 'worst_texture',
        'worst perimeter': 'worst_perimeter',
        'worst area': 'worst_area',
        'worst smoothness': 'worst_smoothness',
        'worst compactness': 'worst_compactness',
        'worst concavity': 'worst_concavity',
        'worst concave points': 'worst_concave_points',
        'worst symmetry': 'worst_symmetry',
        'worst fractal dimension': 'worst_fractal_dimension'
    }
    
    sample_data = {}
    for orig_name, new_name in feature_mapping.items():
        sample_data[new_name] = float(sample[orig_name])
    
    return sample_data, data.target[sample_idx]

# Function to predict using model
def make_prediction(feature_values):
    """Make prediction using the loaded model"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([feature_values])
        
        # Map to original column names (with spaces)
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
        
        # Rename columns
        df_mapped = df.rename(columns=column_mapping)
        
        # Make prediction
        prediction = model.predict(df_mapped)[0]
        probabilities = model.predict_proba(df_mapped)[0]
        
        return {
            'prediction': int(prediction),
            'probabilities': {
                'malignant': float(probabilities[0]),
                'benign': float(probabilities[1])
            },
            'confidence': float(max(probabilities))
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Main app logic
if input_method == "Manual Input":
    st.subheader("🖱️ Manual Feature Input")
    st.info("Enter the breast cancer features manually using the sliders and inputs below.")
    
    feature_values = create_feature_inputs()

elif input_method == "Load Sample Data":
    st.subheader("📂 Sample Data from Dataset")
    st.info("Loading a random sample from the breast cancer dataset.")
    
    if st.button("🔄 Load New Sample"):
        sample_data, true_label = load_sample_data()
        for key, value in sample_data.items():
            st.session_state[f"input_{key}"] = value
        st.session_state['true_label'] = true_label
        st.rerun()
    
    # Show current sample info
    if 'true_label' in st.session_state:
        true_diagnosis = "Benign" if st.session_state['true_label'] == 1 else "Malignant"
        st.success(f"📋 Loaded sample with true diagnosis: **{true_diagnosis}**")
    
    feature_values = create_feature_inputs()

elif input_method == "Random Sample":
    st.subheader("🎲 Random Sample")
    st.info("Generate random feature values within realistic ranges.")
    
    if st.button("🎯 Generate Random Sample"):
        random_values = {}
        for group_features in FEATURE_GROUPS.values():
            for feature_name, config in group_features.items():
                random_val = np.random.uniform(config["min"], config["max"])
                random_values[feature_name] = round(random_val, 4)
                st.session_state[f"input_{feature_name}"] = random_val
        st.rerun()
    
    feature_values = create_feature_inputs()

elif input_method == "API Test":
    st.subheader("🔗 API Testing")
    st.info("Test the FastAPI endpoint directly from the web interface.")
    
    # API configuration
    api_url = st.text_input("API URL", value="http://localhost:8000/predict")
    
    # Use default sample data for API testing
    if st.button("🧪 Test with Sample Data"):
        sample_data = {
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
        
        try:
            response = requests.post(api_url, json=sample_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ API Request Successful!")
                st.json(result)
                
                # Show formatted result
                diagnosis = result.get('diagnosis', 'Unknown')
                confidence = result.get('confidence_score', 0)
                
                if diagnosis == "Benign":
                    st.markdown(f'<div class="prediction-card benign"><h3>✅ Prediction: {diagnosis}</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-card malignant"><h3>⚠️ Prediction: {diagnosis}</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                    
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.text(response.text)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection Error: {str(e)}")
            st.info("Make sure the FastAPI server is running on the specified URL.")
    
    # Manual API testing
    feature_values = create_feature_inputs()

# Prediction section (for non-API methods)
if input_method != "API Test":
    st.markdown("---")
    st.subheader("🔮 Prediction Results")
    
    if st.button("🚀 Make Prediction", type="primary"):
        with st.spinner("Analyzing features..."):
            result = make_prediction(feature_values)
            
            if result:
                prediction = result['prediction']
                probabilities = result['probabilities']
                confidence = result['confidence']
                
                # Display results
                diagnosis = "Benign" if prediction == 1 else "Malignant"
                
                # Main prediction card
                if diagnosis == "Benign":
                    st.markdown(f'''
                    <div class="prediction-card benign">
                        <h2>✅ Prediction: {diagnosis}</h2>
                        <h3>Confidence: {confidence:.2%}</h3>
                        <p>The model predicts this case as <strong>benign (non-cancerous)</strong>.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-card malignant">
                        <h2>⚠️ Prediction: {diagnosis}</h2>
                        <h3>Confidence: {confidence:.2%}</h3>
                        <p>The model predicts this case as <strong>malignant (cancerous)</strong>.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Probability visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Malignant', 'Benign'],
                        values=[probabilities['malignant'], probabilities['benign']],
                        hole=.3,
                        marker_colors=['#ff6b6b', '#51cf66']
                    )])
                    fig_pie.update_layout(
                        title="Prediction Probabilities",
                        showlegend=True,
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=['Malignant', 'Benign'],
                            y=[probabilities['malignant'], probabilities['benign']],
                            marker_color=['#ff6b6b', '#51cf66']
                        )
                    ])
                    fig_bar.update_layout(
                        title="Probability Scores",
                        yaxis_title="Probability",
                        height=400,
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Feature importance visualization (if available)
                st.subheader("📊 Feature Values Overview")
                
                # Group features for better visualization
                feature_df = pd.DataFrame([feature_values]).T
                feature_df.columns = ['Value']
                feature_df['Feature'] = feature_df.index
                feature_df['Group'] = feature_df['Feature'].apply(lambda x: 
                    'Mean Features' if x.startswith('mean_') else
                    'Error Features' if x.endswith('_error') else
                    'Worst Features'
                )
                
                # Create grouped bar chart
                fig_features = px.bar(
                    feature_df, 
                    x='Feature', 
                    y='Value', 
                    color='Group',
                    title="Input Feature Values by Group",
                    height=500
                )
                fig_features.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_features, use_container_width=True)
                
                # Show comparison with true label if available
                if 'true_label' in st.session_state and input_method == "Load Sample Data":
                    true_diagnosis = "Benign" if st.session_state['true_label'] == 1 else "Malignant"
                    is_correct = (prediction == st.session_state['true_label'])
                    
                    if is_correct:
                        st.success(f"✅ **Correct Prediction!** True diagnosis: {true_diagnosis}")
                    else:
                        st.error(f"❌ **Incorrect Prediction.** True diagnosis: {true_diagnosis}")

# Footer
st.markdown("---")
st.markdown("""
### ℹ️ About This Application

This Streamlit web interface provides an interactive way to test the breast cancer prediction model. 
The model uses 30 features derived from digitized images of breast mass to predict whether a tumor is malignant or benign.

**Features include:**
- **Mean features**: Average values of various tumor characteristics
- **Error features**: Standard error of the measurements  
- **Worst features**: Mean of the three largest values

**⚠️ Disclaimer:** This tool is for educational and research purposes only. Always consult healthcare professionals for medical advice.
""")
