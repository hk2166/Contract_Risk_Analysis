import joblib
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODEL_PATH, RISK_LABELS
from nlp.feature_engineering import load_vectorizer, transform_new_text
from nlp.preprocessing import preprocess_text

class ContractRiskAI:
    """
    This class is the 'interface' for our AI. 
    You give it a sentence, and it tells you the risk.
    """
    def __init__(self):
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = load_vectorizer()
        except:
            self.model = None
            self.vectorizer = None

    def analyze_clause(self, text):
        """
        Returns the risk label, confidence percentage, and the 'reasoning' words.
        """
        if self.model is None or self.vectorizer is None:
            self.load_model()
            if self.model is None:
                return "Not Trained", 0.0, []
            
        # 1. Clean the text using the same logic as training
        clean = preprocess_text(text)
        
        # 2. Turn into numbers (the same TF-IDF dictionary)
        features = transform_new_text([clean], self.vectorizer)
        
        # 3. Predict the label
        label_idx = self.model.predict(features)[0]
        label_str = RISK_LABELS.get(label_idx, "Unknown")
        
        # 4. Probability (Confidence)
        probs = self.model.predict_proba(features)[0]
        confidence = np.max(probs)
        
        # 5. Explainability Layer: Why did it choose this?
        # We look at which words in this clause have the highest weights 
        # in our model for the 'High Risk' class.
        explain_data = self.get_explainability(features)
        
        return label_str, confidence, explain_data

    def get_explainability(self, features):
        """
        Identifies the feature names (words) that most heavily influenced
        the risk prediction for this specific clause.
        """
        if self.model is None or not hasattr(self.model, "coef_"):
            return []

        # Get the feature weights for 'High Risk' (Class 1)
        weights = self.model.coef_[0]
        
        # Get feature indices for the non-zero values in this clause
        feature_indices = features.indices
        
        # Map indices to (weight, word)
        feature_names = self.vectorizer.get_feature_names_out()
        
        reasons = []
        for idx in feature_indices:
            weight = weights[idx]
            if weight > 0: # Focus on words that INCREASE risk
                reasons.append((weight, feature_names[idx]))
        
        # Sort by weight importance and return top 5
        reasons.sort(key=lambda x: x[0], reverse=True)
        return [word for weight, word in reasons[:5]]

# Easy-to-use instance
risk_engine = ContractRiskAI()
