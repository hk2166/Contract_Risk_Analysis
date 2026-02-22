import string
import re

# We use spaCy because it's like a'linguistic expert' for Python.
# It understands that "running" and "ran" are the same word ("run").
try:
    import spacy
    try:
        # 'sm' stands for 'small' - it's fast and enough for our needs.
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except (ImportError, Exception):
    nlp = None

def preprocess_text(text):
    """
    This function cleans raw text so our AI can understand it more easily.
    Imagine it as washing and chopping vegetables before cooking!
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase: "Contract" and "contract" should be treated the same.
    text = text.lower()
    
    # 2. Punctuation removal: Commas and periods often don't help the AI detect risk.
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Extra space removal: Clean up messy formatting.
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Lemmatization (using spaCy):
    # This turns words like "agreements" into "agreement".
    if nlp:
        try:
            doc = nlp(text)
            # We also ignore 'stopwords' (common words like 'the', 'is' that add no value).
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
            return " ".join(tokens)
        except Exception:
            pass
            
    # Fallback: If spaCy isn't working, we do a simple split.
    tokens = [t for t in text.split() if len(t) > 2]
    
    return " ".join(tokens)

if __name__ == "__main__":
    # Test our 'cleaning machine'
    sample = "The parties are currently agreeing to the terms!"
    print(f"Before: {sample}")
    print(f"After : {preprocess_text(sample)}")
