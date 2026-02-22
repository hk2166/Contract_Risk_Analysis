import re

def segment_clauses(text):
    """
    Split contract text into clauses based on newlines, numbering, and punctuation.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split by common clause markers:
    # 1. Numbered sections (e.g., 1., 1.1, Article I)
    # 2. Strong punctuation (semi-colons or periods followed by capitals)
    
    # This regex looks for:
    # - Newlines (normalized to space already, so we look for patterns)
    # - Numbered list items like "1. ", "(a) ", etc.
    # - Capitalized headers
    
    # For a robust classical approach, we can split by periods followed by spaces and capitals,
    # or by specific legal markers.
    
    # Simple rule-based split for trial:
    clauses = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\d\.)\s+|(?<=[;])\s+', text)
    
    # Clean up
    clauses = [c.strip() for c in clauses if len(c.strip()) > 10]
    
    return clauses

if __name__ == "__main__":
    test_text = "1. This is the first clause. 2. This is the second clause; it has a semicolon. 3. Final clause here."
    segments = segment_clauses(test_text)
    for i, s in enumerate(segments):
        print(f"[{i}] {s}")
