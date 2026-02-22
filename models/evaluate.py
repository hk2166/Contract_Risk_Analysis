import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.preprocessing import preprocess_text
from config.settings import DATA_DIR, MODEL_PATH, VECTORIZER_PATH

def evaluate_model():
    # Load dataset
    data_path = os.path.join(DATA_DIR, 'contract_clauses.csv')
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Load Model and Vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model or Vectorizer not found. Please run train.py first.")
        return
        
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Preprocess and Transform
    processed_text = df['clause_text'].apply(preprocess_text)
    X = vectorizer.transform(processed_text)
    y_true = df['risk_level']
    
    # Predict
    y_pred = clf.predict(X)
    
    # Results
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()
