import streamlit as st
import pandas as pd
import pdfplumber
import os
import sys

# Making sure the app can find our nlp and models folders.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.clause_segmenter import segment_clauses
from models.inference import risk_engine
from config.settings import RISK_COLORS

# --- PAGE CONFIG ---
st.set_page_config(page_title="Legal AI - Contract Risk", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .main { padding: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚖️ Legal Guard AI")
    st.info("Milestone 1: Logic-based Risk Parser")
    
    st.markdown("---")
    st.write("**How to use:**")
    st.write("1. Upload a contract (PDF or TXT).")
    st.write("2. Wait for the AI to 'read' the clauses.")
    st.write("3. Review the risks flagged in red/green.")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"])

# --- MAIN PAGE ---
st.title("🔍 Intelligent Contract Risk Dashboard")
st.write("Acting as your digital paralegal, this tool scans legal text to find potential dangers.")

def get_text_from_file(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        return file.getvalue().decode("utf-8")

if uploaded_file:
    # 1. Processing
    raw_text = get_text_from_file(uploaded_file)
    clauses = segment_clauses(raw_text)
    
# --- 2. CONTRACT OVERVIEW PANEL ---
    st.subheader("📊 Contract Risk Summary")
    
    results = []
    for c in clauses:
        label, conf, reasons = risk_engine.analyze_clause(c) # UPDATED: Now returns reasons
        results.append({
            "Clause": c, 
            "Risk Level": label, 
            "Confidence": conf,
            "Keywords": reasons
        })
    
    df_results = pd.DataFrame(results)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clauses Scanned", len(clauses))
    with col2:
        high_risk_count = len(df_results[df_results['Risk Level'] == "High Risk"])
        st.metric("High Risk Flags 🚩", high_risk_count)
    with col3:
        avg_conf = df_results['Confidence'].mean()
        st.metric("Model Trust Score", f"{avg_conf:.1%}")

    # --- 3. RISK ANALYTICS OVERVIEW ---
    with st.expander("📈 Advanced Risk Analytics", expanded=False):
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.write("**Risk Distribution**")
            fig_bar = plt.figure(figsize=(6, 4))
            sns.countplot(data=df_results, x='Risk Level', palette="magma", hue='Risk Level', legend=False)
            st.pyplot(fig_bar)
            
        with chart_col2:
            st.write("**Probability Confidence**")
            fig_conf = plt.figure(figsize=(6, 4))
            sns.histplot(df_results['Confidence'], kde=True, color="purple")
            st.pyplot(fig_conf)

    st.markdown("---")

    # --- 4. DETAILED CLAUSE ANALYSIS & EXPLAINABILITY ---
    st.subheader("🔍 Clause-by-Clause AI Audit")
    st.info("The AI provides a 'Mentor Note' for every clause, explaining the specific linguistic triggers it found.")
    
    for i, row in df_results.iterrows():
        label = row['Risk Level']
        color = RISK_COLORS.get(label, "gray")
        icon = "🔴" if label == "High Risk" else "🟢"
        
        with st.expander(f"{icon} {label} ({row['Confidence']:.1%}) - Clause {i+1}"):
            st.markdown(f"**Extracted Text:**")
            st.code(row['Clause'], language="text")
            
            # --- MODEL EXPLAINABILITY PANEL ---
            if label == "High Risk":
                st.warning(f"**AI Reasoning:** This clause was flagged as High Risk because it contains significant legal triggers.")
                if row['Keywords']:
                    st.write(f"**Key Danger Words Detected:**")
                    st.write(", ".join([f"`{w}`" for w in row['Keywords']]))
                st.markdown(f"> *\"This clause may be risky because it contains strong liability or obligation language associated with '{row['Keywords'][0] if row['Keywords'] else 'legal triggers'}'.\"*")
            else:
                st.success("**AI Reasoning:** This clause appears to be standard operational language with low legal liability.")

    st.markdown("---")

    # --- 5. DATASET & TRAINING INSIGHTS ---
    st.subheader("🧠 Model Explainability & Health")
    insight_tab1, insight_tab2 = st.tabs(["AI Foundations", "Training Artifacts"])
    
    with insight_tab1:
        st.write("**How this AI thinks:**")
        st.write("- **Model**: Balanced Logistic Regression")
        st.write("- **Linguistic Engine**: spaCy Lemmatization")
        st.write("- **Feature Strategy**: TF-IDF (N-gram 1,2)")
        st.markdown("""
        > This project uses **Supervised Learning**. The AI was trained on a dataset called `legal_docs_modified.csv`, 
        > learning the difference between standard business language and risky legal obligations.
        """)

    with insight_tab2:
        try:
            # We display the confusion matrix we generated during training!
            st.write("**Model Accuracy Matrix (The Final Exam)**")
            st.image("artifacts/logistic_regression_matrix.png", caption="Evaluation Heatmap: Precision vs Recall")
        except:
            st.write("Training artifacts not found. Please run `python3 models/train.py`.")

else:
    st.warning("Please upload a file in the sidebar to begin the analysis.")
    
    # Example table for beginner users
    st.markdown("### 💡 What the AI looks for:")
    example_data = {
        "Clause Type": ["Termination", "Confidentiality", "Payment"],
        "Risk Example": ["'Either party can end with 0 days notice'", "'All info is public'", "'Interest rate is 50%'"],
        "Expected Risk": ["🔴 High", "🔴 High", "🟡 Medium"]
    }
    st.table(example_data)
