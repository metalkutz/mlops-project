import logging
import yaml
import mlflow
import mlflow.sklearn
from steps.ingest import Ingestion
from steps.preprocessing import Preprocess_data
from steps.train import Trainer
from steps.predict import Predictor
from sklearn.metrics import classification_report

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    # Load data
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    logging.info("Data ingestion completed successfully")

    # Preprocess data
    preprocessor = Preprocess_data()
    train_data = preprocessor.preprocess_data(train)
    test_data = preprocessor.preprocess_data(test)
    logging.info("Data preprocessing completed successfully")

    # Prepare and train model
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logging.info("Model training completed successfully")

    # Evaluate model
    predictor = Predictor()
    X_test, y_test = predictor.feature_target_separator(test_data)
    accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
    logging.info("Model evaluation completed successfully")
    
    # Print evaluation results
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
    print(f"\n{class_report}")
    print("=====================================================\n")


def train_with_mlflow():

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Model Training Experiment")
    
    with mlflow.start_run() as run:
        # Load data
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        # Preprocess data
        preprocessor = Preprocess_data()
        train_data = preprocessor.preprocess_data(train)
        test_data = preprocessor.preprocess_data(test)
        logging.info("Data preprocessing completed successfully")

        # Prepare and train model
        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logging.info("Model training completed successfully")
        
        # Evaluate model
        predictor = Predictor()
        X_test, y_test = predictor.feature_target_separator(test_data)
        accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
        report = classification_report(y_test, trainer.pipeline.predict(X_test), output_dict=True)
        logging.info("Model evaluation completed successfully")
        
        # Tags 
        mlflow.set_tag('Model developer', 'metalkutz')
        mlflow.set_tag('preprocessing', 'Standard Scaler')
        
        # Log metrics
        model_params = config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("roc", float(roc_auc_score))
        mlflow.log_metric('precision', float(report['weighted avg']['precision']))
        mlflow.log_metric('recall', float(report['weighted avg']['recall']))
        # Provide an input_example for signature inference
        input_example = X_test[:5].copy()
        # Convert integer columns to float64 to avoid schema enforcement errors
        for col in input_example.select_dtypes(include='int').columns:
            input_example[col] = input_example[col].astype('float64')
        mlflow.sklearn.log_model(trainer.pipeline, "model", input_example=input_example)
                
        # Register the model using the 'name' parameter
        model_name = "breast_cancer_model" 
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, name=model_name)

        logging.info("MLflow tracking completed successfully")

        # Print evaluation results
        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
        print(f"\n{class_report}")
        print("=====================================================\n")
        
if __name__ == "__main__":
    # main()
    train_with_mlflow()