# Sample data extraction file which loads the breast cancer dataset from sklearn.datasets
import pandas as pd
import os
from sklearn.datasets import load_breast_cancer

def extract_data():
    if not os.path.exists("data"):
        os.mkdir("data")
    
    # Load the breast cancer dataset
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    
    # Create DataFrame with feature names
    df = pd.DataFrame(X, columns=breast_cancer['feature_names'])
    df['target'] = y
    
    # Split data into train and test sets (80-20 split)
    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Save to CSV files
    train_data.to_csv("data/train.csv", index=False)
    test_data.to_csv("data/test.csv", index=False)
    
    # Create a production dataset (same as test for this example)
    test_data.to_csv("data/production.csv", index=False)

    print(f"Extracted breast cancer dataset successfully")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Features: {len(breast_cancer.feature_names)}")
    print(f"Target classes: {list(breast_cancer.target_names)}")

if __name__ == "__main__":
    extract_data()