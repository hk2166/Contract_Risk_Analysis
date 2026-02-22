# Intelligent Contract Risk Analysis and Agentic Legal Assistance System

> **Milestone 1: Automated Clause Classification and Risk Assessment Engine**

---

## 📄 Project Overview

The **Intelligent Contract Risk Analysis System** is a production-grade AI platform designed to automate the review of legal documents. By leveraging classical Natural Language Processing (NLP) and Supervised Machine Learning, the system decomposes complex legal contracts into individual clauses, identifies high-risk provisions, and provides real-time confidence scores. This project establishes the foundation for a scalable, agentic legal assistance framework by focusing on robust feature engineering and modular architecture.

## ✨ Key Features

- **Automated Clause Segmentation**: Intelligent parsing of PDF and TXT documents into discrete legal clauses based on structural markers (Articles, Sections, Numbering).
- **Risk Classification Engine**: High-performance classification into **High Risk** and **Low Risk** categories using real-world legal datasets.
- **Linguistic Preprocessing**: Advanced pipeline using `spaCy` for lemmatization, custom legal stopword filtering, and text normalization.
- **Interactive Dashboard**: A professional Streamlit-based UI for contract uploading, real-time highlighting, and risk distribution analytics.
- **Modular Architecture**: Clean, enterprise-ready codebase organized for scalability and maintainability.

## 🏗️ System Architecture

The platform follows a layered NLP pipeline designed for efficiency and transparency:

1.  **Ingestion Layer**: Text extraction from unstructured PDF/TXT files using `pdfplumber`.
2.  **Segmentation Layer**: Regex-driven logic identifies legal boundaries to isolate clauses.
3.  **Preprocessing Layer**: spaCy-powered lemmatization and noise reduction.
4.  **Feature Layer**: TF-IDF Vectorization with Bi-gram support to capture legal context.
5.  **Inference Layer**: Logistic Regression model predicts risk probability and labels.
6.  **Presentation Layer**: Streamlit dashboard visualizes the analysis with color-coded alerts.

## 🛠️ Tech Stack

| Component            | Technology                          |
| :------------------- | :---------------------------------- |
| **Language**         | Python 3.9+                         |
| **NLP**              | spaCy, Scikit-learn                 |
| **Machine Learning** | Logistic Regression, Decision Trees |
| **Data Handling**    | Pandas, Numpy                       |
| **UI Framework**     | Streamlit                           |
| **Parsing**          | pdfplumber                          |
| **Serialization**    | Joblib                              |

## 📊 Dataset & Model Details

### Dataset

The system is trained on the `legal_docs_modified.csv` dataset, containing thousands of labeled legal clauses across various categories (Interest, Termination, Indemnification, etc.).

- **Features**: `clause_text`
- **Target**: `clause_status` (Mapping: `0: Low Risk`, `1: High Risk`)

### Model Specification

- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
  - _N-gram Range_: (1, 2)
  - _Max Features_: 5,000
- **Classifier**: Logistic Regression (Primary) with balanced class weights.
- **Performance Metrics**:
  - **Precision**: Minimized false positives for risk identification.
  - **Recall**: Optimized to ensure critical clauses are flagged.
  - **F1-Score**: Weighted average showing robust performance across 20+ clause types.

## 🧠 The Core Science Behind the AI

This project isn't just a dashboard; it's a demonstration of fundamental AI/ML and NLP principles. Here's a look under the hood:

### 1. Artificial Intelligence vs. Machine Learning

- **Artificial Intelligence (AI)** is the broad field of creating systems that can perform tasks requiring human intelligence.
- **Machine Learning (ML)** is a subset of AI that focuses on building systems that _learn_ from data rather than following strict, manually coded instructions.

> [!NOTE]
> Instead of writing a million "if-else" statements for legal words, we give the model examples and let it discover the patterns itself.

### 2. Supervised Learning & Classification

This is a **Supervised Learning** project. We "supervise" the AI by giving it a "labeled" dataset (e.g., this sentence = High Risk).

- **The Task**: This is a **Binary Classification** problem. The AI must decide if a clause belongs to one of two classes: `0` (Low Risk) or `1` (High Risk).

### 3. Logistic Regression & The Sigmoid Function

Our primary "brain" is **Logistic Regression**.

- **How it works**: It calculates a weighted sum of the words in a clause to determine a probability.
- **The Math**: It uses the **Sigmoid Function** to squash that probability between 0 and 1. If the score is > 0.5, it's flagged as High Risk.

### 4. Natural Language Processing (NLP) Foundations

To help a computer "read" legalese, we use three key techniques:

- **Tokenization**: Breaking sentences into individual words (tokens).
- **Lemmatization**: Reducing words to their root form (e.g., "indemnifying" -> "indemnify"). This helps the AI treat different versions of a word as the same concept.
- **TF-IDF (The Secret Sauce)**: **Term Frequency-Inverse Document Frequency**. It highlights words that are unique and meaningful (like "Liability") while ignoring common filler words (like "the").

---

## 📁 Project Structure

```bash
contract-risk-ai/
├── data/                    # Dataset storage (legal_docs_modified.csv)
├── analysis/                # EDA scripts and eda.ipynb
├── nlp/                     # Core NLP logic
│   ├── preprocessing.py     # spaCy lemmatization & cleaning
│   ├── clause_segmenter.py  # Regex-based segmentation
│   └── feature_engineering.py # TF-IDF vectorization
├── models/                  # ML Pipeline
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Metrics & Visualization
│   └── inference.py         # Real-time prediction pipeline
├── artifacts/               # Serialized models (classifier.pkl)
├── app/                     # UI Layer
│   └── streamlit_app.py     # Streamlit application
├── config/                  # Configuration & Global settings
└── requirements.txt         # Project dependencies
```

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/suvendukungfu/Intelligent-Contract-Risk-Analysis-and-Agentic-Legal-Assistance-System.git
cd contract-risk-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Training the Model

Generate the model artifacts by running the training script:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 models/train.py
```

### 4. Running the Dashboard

Launch the Streamlit interface locally:

```bash
streamlit run app/streamlit_app.py
```

## 🌐 Public Deployment

Live Demo: [Deployment Link Placeholder]

## 📈 Exploratory Data Analysis (EDA)

Comprehensive analysis of the legal corpus is available in `analysis/eda.ipynb`. Key insights include:

- **High Class Imbalance**: Addressed using balanced class weights in the ML model.
- **Keyword Prominence**: TF-IDF reveals high-weight terms like "Termination", "Salary", and "Liability".
- **Structural Variation**: Large variance in clause length across different legal categories.

## 🔮 Future Scope (Milestone 2 - Agentic AI)

- **Retrieval-Augmented Generation (RAG)**: Connect clauses to external legal knowledge bases.
- **Multi-Agent Negotiation**: Simulate document revisions between "Owner" and "Vendor" agents.
- **Semantic Consistency**: Use LLMs to check if "Definitions" match their usage throughout the document.

## ⚖️ Legal Disclaimer

_This system is an AI research project intended for informational purposes only. It does not constitute legal advice. The classifications and risk scores are generated based on mathematical patterns in the training data and should be reviewed by a qualified legal professional._

---

**Maintained by**: [Suvendu kumar Sahoo](https://github.com/suvendukungfu)
