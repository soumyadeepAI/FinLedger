# FinLedger - Hybrid Bank Transaction Classifier & Anomaly Detector

This is my **First-Semester Machine Learning Project**. It categorizes Indian bank transactions (`UPI`, `POS`, `NEFT`, `IMPS`, `ATM`) into 12 spending categories and detects unusual spending spikes.

## Architecture (Method 2: Two-Stage Hybrid Pipeline)
1. **Stage 1 (Known-Merchant Prior Dictionary):** Checks if a transaction description contains a well-known brand or app (`ZOMATO`, `SWIGGY`, `FLIPKART`, `UBER`, `ZERODHA`, etc.). If matched, it assigns the category immediately with 100% certainty.
2. **Stage 2 (Word + Character N-Gram TF-IDF + Logistic Regression):** Any local shop or unknown merchant not in the dictionary is passed to our Scikit-Learn pipeline combining word n-grams `(1, 2)` and character subwords `char_wb (3, 5)`.
3. **Confidence Thresholding:** If Stage 2's highest predicted class probability is below `45%`, the row is flagged as `Needs_Review = True`.
4. **Context-Aware Isolation Forest:** Detects unusual high-spending anomalies using a 2D feature space (`log_debit` and `category_deviation` above the predicted category's median spend) so cheap purchases are never falsely flagged.

## Project Structure
- `01_train_the_model.ipynb`: Step-by-step Jupyter Notebook that trains, evaluates (including the Unseen-Merchant `GroupShuffleSplit` benchmark), explains model weights, and exports `model.joblib`.
- `app.py`: Streamlit web dashboard for uploading bank statements, viewing category charts, and inspecting anomaly alerts.
- `model.joblib`: Pre-trained hybrid artifact (Dictionary + Word/Char TF-IDF + Logistic Regression pipeline).
- `data/demo_training.csv`: 5,500 labeled Indian bank transactions across 12 categories.
- `data/sample_statement.csv`: 20-row sample bank statement for testing the app and anomaly detector.
- `requirements.txt`: Python package dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run `01_train_the_model.ipynb` in Jupyter Notebook / VS Code / Google Colab, or launch the web app directly:
   ```bash
   streamlit run app.py
   ```
