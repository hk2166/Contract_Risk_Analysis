import pandas as pd
import os

path = "/Users/suvendusahoo/Desktop/sem4aiproject /contract-risk-ai/data/legal_docs_modified.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    print("\nSample Data:")
    print(df.head())
    print("\nClause Status Distribution:")
    print(df['clause_status'].value_counts().head(20))
    print("\nClause Type Distribution:")
    print(df['clause_type'].value_counts().head(20))
else:
    print(f"File not found at {path}")
