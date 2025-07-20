import os
import joblib
import yaml

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def create_pipeline(self):
        """
        Create a machine learning pipeline for the breast cancer dataset.
        Since the data is already preprocessed (standardized), we only need
        SMOTE for class balancing and the classifier.
        """
        # No preprocessing needed as data is already standardized by preprocessing step
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        
        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'XGBClassifier': XGBClassifier
        }
    
        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)

        # Simplified pipeline without preprocessing (already done in preprocessing step)
        pipeline = Pipeline([
            ('smote', smote),
            ('model', model)
        ])

        return pipeline

    def feature_target_separator(self, data):
        """
        Separate features and target from the breast cancer dataset.
        All columns except 'target' are features.
        """
        # Get all feature columns (everything except 'target')
        feature_columns = [col for col in data.columns if col != 'target']
        X = data[feature_columns]
        y = data['target']
        return X, y

    def train_model(self, X_train, y_train):
        """
        Train the model pipeline on the breast cancer dataset.
        """
        print(f"Training {self.model_name} on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        self.pipeline.fit(X_train, y_train)
        print("Model training completed successfully")

    def save_model(self):
        """
        Save the trained model pipeline to disk.
        """
        # Create the model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        joblib.dump(self.pipeline, model_file_path)
        print(f"Model saved to: {model_file_path}")