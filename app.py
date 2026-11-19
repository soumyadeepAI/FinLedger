from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def clean_merchant_text(description):
    """Strips payment channel prefixes (UPI, P2M, POS DEBIT) and long reference IDs so ML only sees the merchant words."""
    text = str(description).upper()
    text = re.sub(r'[/\-_]+', ' ', text)
    text = re.sub(r'\b\d{4,}\b', ' ', text)
    text = re.sub(r'\b(UPI|P2M|P2A|POS|DEBIT|IMPS|ACH|REF|ONLINE|ORDER|PAYMENT|TXN|NO)\b', ' ', text)
    return ' '.join(text.split()).strip() or 'UNKNOWN'


def rule_based_lookup(description, merchant_dict):
    """Stage 1: Check if the transaction matches any well-known brand in our dictionary."""
    text = str(description).upper()
    clean_tokens = set(re.sub(r'[^A-Z0-9]+', ' ', text).split())
    for category, keywords in merchant_dict.items():
        for kw in keywords:
            if ' ' in kw:
                if kw in text:
                    return category
            else:
                if kw in clean_tokens or (len(kw) >= 4 and any(tok.startswith(kw) for tok in clean_tokens)):
                    return category
    return None


def analyze_statement_df(df_input, artifact):
    """
    Runs the complete FinLedger Two-Stage Hybrid Classifier + Context-Aware Isolation Forest
    on a bank statement DataFrame containing at least a 'Description' column (and optional 'Debit'/'Credit').
    """
    df = df_input.copy()
    
    col_map = {c.lower().strip(): c for c in df.columns}
    desc_col = col_map.get('description', df.columns[0])
    debit_col = col_map.get('debit', None)
    credit_col = col_map.get('credit', None)
    
    trained_ml_model = artifact['model']
    merchant_dict = artifact.get('merchant_dict', {})
    conf_threshold = float(artifact.get('conf_threshold', 0.45))
    
    descriptions = df[desc_col].astype(str).tolist()
    cleaned_descriptions = [clean_merchant_text(d) for d in descriptions]
    classes = trained_ml_model.classes_
    
    ml_probs = trained_ml_model.predict_proba(cleaned_descriptions)
    ml_best_idx = np.argmax(ml_probs, axis=1)
    ml_best_cat = classes[ml_best_idx]
    ml_best_conf = np.max(ml_probs, axis=1)
    
    categories, confidences, methods, needs_review = [], [], [], []
    for i, text in enumerate(descriptions):
        rule_cat = rule_based_lookup(text, merchant_dict)
        if rule_cat is not None:
            categories.append(rule_cat)
            confidences.append(1.00)
            methods.append('Stage 1: Dictionary')
            needs_review.append(False)
        else:
            conf = float(ml_best_conf[i])
            categories.append(str(ml_best_cat[i]))
            confidences.append(round(conf, 3))
            methods.append('Stage 2: ML (Word+Char TF-IDF)')
            needs_review.append(bool(conf < conf_threshold))
            
    df['Predicted_Category'] = categories
    df['Confidence'] = confidences
    df['Method'] = methods
    df['Needs_Review'] = needs_review
    
    # Context-Aware Anomaly Detection (2D Isolation Forest: log_debit + category_deviation)
    df['Unusual_Spend'] = False
    df['Category_Median'] = 0.0
    
    if debit_col is not None:
        df[debit_col] = pd.to_numeric(df[debit_col], errors='coerce').fillna(0.0)
        debit_mask = df[debit_col] > 0
        if debit_mask.sum() >= 5:
            debits = df.loc[debit_mask].copy()
            cat_medians = debits.groupby('Predicted_Category')[debit_col].transform('median')
            df.loc[debit_mask, 'Category_Median'] = cat_medians
            
            log_amt = np.log1p(debits[debit_col])
            log_med = np.log1p(cat_medians)
            cat_dev = log_amt - log_med
            
            X_anom = pd.DataFrame({
                'log_debit': log_amt,
                'cat_deviation': cat_dev
            })
            iso = IsolationForest(contamination=0.12, random_state=42)
            raw_unusual = (iso.fit_predict(X_anom) == -1)
            df.loc[debit_mask, 'Unusual_Spend'] = (raw_unusual & (cat_dev > 0.5)).values
            
    if credit_col is not None:
        df[credit_col] = pd.to_numeric(df[credit_col], errors='coerce').fillna(0.0)
        
    return df


def run_streamlit_ui():
    import streamlit as st
    
    st.set_page_config(page_title="FinLedger - ML Expense Categorizer", layout="wide")
    st.title("FinLedger: Hybrid Bank Transaction Classifier & Anomaly Detector")
    st.caption("First-Semester ML Project | Stage 1: Known-Merchant Dictionary + Stage 2: Cleaned Word & Character N-Gram TF-IDF + Logistic Regression")

    model_path = Path("model.joblib")
    if not model_path.exists():
        st.error("model.joblib not found! Please run all cells in 01_train_the_model.ipynb first.")
        st.stop()

    artifact = joblib.load(model_path)

    with st.sidebar:
        st.header("Model Card & Metrics")
        st.write(f"**Training Source:** {artifact.get('source', 'Unknown')}")
        st.metric("Standard Test Accuracy", f"{artifact.get('accuracy', 0.0)*100:.1f}%")
        st.metric("Standard Macro F1", f"{artifact.get('macro_f1', 0.0):.3f}")
        if 'unseen_accuracy' in artifact:
            st.metric("Unseen-Merchant Accuracy", f"{artifact['unseen_accuracy']*100:.1f}%")
            st.metric("Unseen-Merchant Macro F1", f"{artifact['unseen_macro_f1']:.3f}")
        st.divider()
        st.subheader("Quick Single-Transaction Test")
        test_input = st.text_input("Enter a transaction description:", "UPI/P2M/839201/ZOMATO")
        if test_input:
            single_df = analyze_statement_df(pd.DataFrame({'Description': [test_input]}), artifact)
            row = single_df.iloc[0]
            st.success(f"**Category:** {row['Predicted_Category']}")
            st.caption(f"Handled by: `{row['Method']}` | Confidence: `{row['Confidence']}`")
            if row['Needs_Review']:
                st.warning("Low confidence prediction — flagged for human review.")

    st.subheader("1. Upload or Select a Bank Statement CSV")
    uploaded_file = st.file_uploader("Upload a CSV with columns: Date, Description, Debit, Credit", type=["csv"])
    
    sample_path = Path("data/sample_statement.csv")
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.info("Loaded uploaded CSV file.")
    elif sample_path.exists():
        raw_df = pd.read_csv(sample_path)
        st.info("Showing built-in demo file (`data/sample_statement.csv`). Upload your own CSV above to test!")
    else:
        st.warning("Please upload a CSV file to begin.")
        st.stop()

    analyzed_df = analyze_statement_df(raw_df, artifact)

    total_debit = analyzed_df['Debit'].sum() if 'Debit' in analyzed_df.columns else 0.0
    total_credit = analyzed_df['Credit'].sum() if 'Credit' in analyzed_df.columns else 0.0
    unusual_count = int(analyzed_df['Unusual_Spend'].sum())
    review_count = int(analyzed_df['Needs_Review'].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend (Debit)", f"₹{total_debit:,.2f}")
    col2.metric("Total Income (Credit)", f"₹{total_credit:,.2f}")
    col3.metric("Unusual High Spends Flagged", unusual_count)
    col4.metric("Low-Confidence (Needs Review)", review_count)

    st.subheader("2. Categorized Statement & Anomaly Flags")
    st.dataframe(analyzed_df, use_container_width=True)

    if 'Debit' in analyzed_df.columns:
        st.subheader("3. Spending Breakdown by Category")
        spend_by_cat = (
            analyzed_df[analyzed_df['Debit'] > 0]
            .groupby('Predicted_Category')['Debit']
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(spend_by_cat)

    if unusual_count > 0:
        st.subheader("4. Unusual Spending Alerts (Context-Aware Isolation Forest)")
        st.warning("These transactions were significantly higher than the typical median spend for their category:")
        st.dataframe(analyzed_df[analyzed_df['Unusual_Spend']], use_container_width=True)

    if review_count > 0:
        st.subheader("5. Transactions Needing Manual Review (Confidence < 45%)")
        st.dataframe(analyzed_df[analyzed_df['Needs_Review']], use_container_width=True)

    csv_bytes = analyzed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Categorized Statement CSV",
        data=csv_bytes,
        file_name="finledger_categorized_statement.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    run_streamlit_ui()
