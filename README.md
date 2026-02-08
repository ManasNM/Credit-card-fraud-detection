# Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview
This project focuses on building an end-to-end **Credit Card Fraud Detection system** using machine learning techniques.  
The primary objective is to accurately identify fraudulent transactions in a **highly imbalanced dataset**, while minimizing false positives and financial risk.

The project follows a structured data science lifecycle:
- Data understanding and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Model building and evaluation
- Threshold tuning
- Business insights and recommendations

A strong emphasis is placed on **recall-focused evaluation**, as missing fraudulent transactions can lead to significant financial losses.

---

## 🧠 Business Problem
Credit card fraud is rare but extremely costly.  
The key challenges include:
- Extreme class imbalance (fraud < 0.2%)
- Fraud patterns constantly evolving
- Need to balance fraud detection with customer experience

This project aims to:
- Detect fraudulent transactions proactively
- Maximize fraud recall
- Minimize false positives
- Build a scalable and deployable ML solution

---

## 📂 Project Contents
This repository contains the following components:

- **Jupyter Notebook**
  - Complete end-to-end analysis
  - Data preprocessing
  - EDA and visualization
  - Feature engineering
  - Model training and evaluation
  - Threshold tuning

- **Technical Documentation (MS Word)**
  - Detailed technical explanation of methodology
  - Preprocessing, modeling, evaluation, and business impact

- **Presentation Deck (PowerPoint)**
  - Stakeholder-ready slides
  - Visual storytelling of insights and model performance

- **Dataset (CSV)**
  - Credit card transaction dataset used for modeling

---

## 📥 Dataset Information
- Source: European cardholders credit card transactions
- Transactions: 284,807
- Fraud cases: 492 (~0.17%)
- Features:
  - `V1–V28`: PCA-transformed features (anonymized)
  - `Amount`: Transaction amount
  - `Time`: Seconds elapsed between transactions
  - `Class`: Target variable (0 = Legitimate, 1 = Fraud)

### 🔗 Dataset Download Link
You can download the dataset from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


---

## 🔍 Step 1: Data Understanding & Validation
- Verified dataset structure and schema
- Checked data types for all features
- Identified target variable (`Class`)
- Removed **1,081 duplicate records**
- Ensured structural consistency after cleaning

✅ Result: Clean and reliable dataset for further analysis

---

## 🧹 Step 2: Data Preprocessing & Feature Engineering
- Confirmed **absence of missing values**
- Scaled numerical features (`Amount`, `Time`) using StandardScaler
- Retained PCA features (`V1–V28`) without interpretability assumptions
- Addressed severe class imbalance using:
  - **SMOTE oversampling**
  - **Class-weight adjustments**
- Performed **stratified train-test split** to preserve class distribution

---

## 📊 Step 3: Exploratory Data Analysis (EDA)
### Univariate Analysis
- Distribution of `Amount`, `Time`, and `Class`
- Severe class imbalance visualized clearly

### Bivariate Analysis
- Fraud vs non-fraud transaction characteristics
- Transaction amount anomalies in fraudulent cases
- Temporal concentration of fraud events

### Time-Based Analysis
- Fraud patterns observed at specific transaction time windows

### Correlation Analysis
- Correlations analyzed only for non-PCA features (`Amount`, `Time`)
- PCA components intentionally excluded from interpretation

---

## 🚨 Outlier Analysis
- Detected outliers using **IQR method**
- Significant outliers found in transaction amounts
- No meaningful outliers detected in time feature
- Outliers were **flagged, not removed**, recognizing that extreme values may represent genuine fraud

📌 Key Principle:  
> In fraud detection, extreme behavior is often the signal — not noise.

---

## 🤖 Step 4: Model Building & Evaluation
### Baseline Model
- **Logistic Regression**
- Used as performance benchmark
- High ROC-AUC and strong recall

### Advanced Models
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

These models were selected to:
- Capture non-linear fraud patterns
- Improve fraud recall
- Reduce false negatives

---

## 📈 Evaluation Metrics
Fraud detection requires specialized evaluation metrics:

- **Recall (Primary Metric)**  
  → Measures ability to catch fraudulent transactions
- Precision
- F1-Score
- ROC-AUC
- Confusion Matrix

📌 Accuracy was intentionally not prioritized due to class imbalance.

---

## ⚖️ Threshold Tuning
- Decision thresholds were adjusted to analyze:
  - Fraud recall vs false positives
- Business-driven trade-off evaluation performed
- Final threshold selected to minimize financial risk rather than maximize accuracy

---

## 🏆 Model Comparison Summary
| Model | ROC-AUC | Recall | Key Observation |
|-----|--------|-------|----------------|
| Logistic Regression | ~0.99 | High | Strong baseline |
| Random Forest | ~1.00 | Very High | Best overall performance |
| Gradient Boosting | ~0.998 | High | Stable and robust |

📌 **Random Forest** was selected as the final model based on risk minimization.

---

## 💡 Key Insights
- Fraudulent transactions often show **amount anomalies**
- Fraud events are **temporally clustered**
- Recall-focused evaluation is critical
- Class imbalance handling dramatically improves detection
- Threshold tuning is essential for real-world deployment

---

## 🏦 Business Recommendations
- Deploy recall-optimized fraud detection models
- Use **threshold-based alert systems** to balance customer experience
- Continuously monitor fraud patterns
- Retrain models periodically to handle evolving fraud tactics
- Integrate model outputs with real-time fraud monitoring systems

---

## 🛠 Tools & Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Jupyter Notebook
- MS Word (Technical Documentation)
- PowerPoint (Stakeholder Presentation)

---

## 👨‍💻 Author
**Manas Nayan Mukherjee**  

---

## 🚀 Final Note
This project demonstrates a **production-oriented fraud detection pipeline**, emphasizing:
- Data integrity
- Business risk minimization
- Practical machine learning decision-making

It is designed to reflect **real-world fintech and banking use cases**, not just academic modeling.
