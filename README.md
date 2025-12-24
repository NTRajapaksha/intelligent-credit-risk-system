# 🏦 Intelligent Credit Risk Scoring Engine & AI Underwriter

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-green) ![Polars](https://img.shields.io/badge/BigData-Polars-red) ![Llama 3.2](https://img.shields.io/badge/GenAI-Llama3.2-purple) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

## 📋 Executive Summary

An **enterprise-grade credit risk modeling system** that combines traditional banking methodologies with modern AI/ML techniques. This project demonstrates the full lifecycle of a lending decision engine—from big data processing to real-time scorecard generation with AI-powered explanations.

**Key Innovation:** Hybrid architecture where LightGBM predicts default probability, traditional actuarial methods calibrate credit scores, and Llama 3.2 generates human-readable underwriting decisions.

### Business Value
- **Automation**: Instantly processes loan applications with 71% AUC accuracy
- **Compliance**: Industry-standard KS statistic (30.5+) meets regulatory requirements
- **Transparency**: AI-generated explanations for every decision improve customer experience
- **Scalability**: Handles 2.2M+ records with <1GB RAM using streaming pipelines

---

## 🎯 Problem Statement

Modern banks face three challenges when building credit risk systems:

1. **Scale**: Traditional pandas-based pipelines crash on multi-million row datasets
2. **Interpretability**: Black-box ML models lack regulatory compliance and customer trust
3. **Automation**: Manual underwriting is slow and inconsistent

**This project solves all three** by combining big data engineering, actuarial science, and generative AI.

---

## 🚀 Key Technical Features

### 1. Big Data Pipeline (MLOps Best Practices)
**Challenge**: Lending Club dataset (2007-2018) contains 2.2M+ loan records—too large for standard in-memory processing.

**Solution**:
```python
# Memory-efficient streaming with Polars
import polars as pl
df = pl.scan_csv("accepted_2007_to_2018Q4.csv.gz")
    .filter(pl.col("loan_status").is_not_null())
    .select(relevant_columns)
    .collect(streaming=True)
```

**Result**: Processes full dataset with <1GB RAM usage vs. 8GB+ for pandas.

### 2. Banking-Grade Feature Engineering

**Weight of Evidence (WoE) Binning**:
- Transforms continuous variables into monotonic risk buckets
- Handles outliers and missing values robustly
- Example: Income ranges mapped to risk scores

**Information Value (IV) Filtering**:
- Automated feature selection retaining only predictive signals
- Filters out noise (IV < 0.02) and keeps strong predictors (IV > 0.3)
- Selected features: `int_rate`, `grade`, `dti`, `annual_inc`, `revol_util`

### 3. LightGBM Risk Model

**Why LightGBM?**
- Handles categorical features natively
- Faster training than XGBoost on tabular data
- Built-in regularization prevents overfitting

**Performance Metrics**:
| Metric | Score | Industry Benchmark |
|--------|-------|-------------------|
| **AUC** | 0.71 | >0.65 (Good) |
| **KS Statistic** | 30.53 | >25 (Deployable) |
| **Avg. Precision** | 0.45 | Context-dependent |

### 4. Actuarial Scorecard Calibration

Converts log-odds into FICO-style scores (300-850) using industry-standard formula:

```
Score = Offset + (Factor × ln(Odds))

Where:
- Target Score: 600 at 50:1 odds
- PDO (Points to Double Odds): 20
- Factor = PDO / ln(2) ≈ 28.85
- Offset = 600 - (28.85 × ln(50)) ≈ 487
```

**Decision Thresholds**:
- **550+**: Auto-Approved (Green)
- **520-549**: Manual Review Required (Yellow)
- **<520**: Rejected (Red)

### 5. Generative AI Integration (Llama 3.2)

**Architecture**: Hybrid AI system separating prediction from explanation.

```python
# Predictive AI: Calculates risk
prob_default = lightgbm_model.predict(woe_features)

# Generative AI: Explains risk
explanation = ollama.chat(
    model='llama3.2',
    messages=[{
        'role': 'user',
        'content': f'Explain this {decision} with {prob_default:.1%} risk...'
    }]
)
```

**Benefits**:
- **Transparency**: Every decision includes a natural language justification
- **Personalization**: Explanations reference specific applicant data
- **Compliance**: Meets fair lending documentation requirements

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Processing** | Polars, Pandas, NumPy | Streaming ETL pipeline |
| **Feature Engineering** | Scorecardpy | WoE binning, IV calculation |
| **Machine Learning** | LightGBM, Scikit-Learn | Gradient boosting classifier |
| **Generative AI** | Ollama, Llama 3.2 | Local LLM for explanations |
| **Web Interface** | Streamlit | Interactive dashboard |
| **Serialization** | Joblib, Pickle | Model persistence |
| **Storage** | Parquet | Columnar data format |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- 8GB+ RAM recommended
- Ollama installed ([Download](https://ollama.ai))

### Step 1: Clone Repository
```bash
git clone https://github.com/NTRajapaksha/intelligent-credit-risk-system.git
cd credit-risk-engine
```

### Step 2: Install Python Dependencies
```bash
pip install polars pandas numpy scikit-learn lightgbm scorecardpy streamlit joblib
```

### Step 3: Install Ollama & Pull Llama 3.2
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the Llama 3.2 model
ollama pull llama3.2
```

### Step 4: Prepare Training Data (Optional)

If you want to retrain the model from scratch:

1. Download the Lending Club dataset: [accepted_2007_to_2018Q4.csv.gz](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Place it in the project root directory
3. Run the training pipeline:

```bash
python train_model.py
```

This will generate:
- `lgb_credit_model.pkl` (Trained LightGBM model)
- `woe_bins.pkl` (WoE transformation rules)

**Note**: Pre-trained models are included in the repository for immediate use.

### Step 5: Launch the Application

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Run Streamlit app
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📊 Using the Dashboard

### Workflow

1. **Enter Applicant Details** (Left Sidebar):
   - Annual Income
   - Debt-to-Income Ratio (DTI)
   - Loan Amount
   - Lending Club Grade (A-G)
   - Interest Rate
   - Employment Length

2. **Calculate Risk Score** (Click Button):
   - System applies WoE transformations
   - LightGBM predicts default probability
   - Score is calibrated to 300-850 range

3. **Review Decision** (Main Panel):
   - Color-coded decision (Green/Yellow/Red)
   - Probability of default percentage
   - AI-generated explanation from Llama 3.2
   - Technical details (expandable)

### Example Output

```
Credit Score: 568
✅ APPROVED
Probability of Default: 5.64%

🤖 AI Underwriter Analysis:
"We are pleased to inform you that your loan application has been approved, contingent upon the satisfactory
completion of our standard credit requirements. Your excellent income of $150,000 and prudent debt-to-income
ratio of 6.3 demonstrate responsible financial management, which significantly mitigates risk in this case.
We congratulate you on achieving a strong credit profile, allowing us to extend a competitive loan offer."
```

---

## 📈 Model Performance Details

### Training Dataset
- **Source**: Lending Club (2007-2018 Q4)
- **Records**: 2,260,668 loans
- **Features**: 151 original columns → 23 engineered features
- **Target**: Binary classification (Default vs. Fully Paid)

### Model Evaluation

**Discrimination Power**:
- **AUC-ROC**: 0.71 (Good separation between classes)
- **KS Statistic**: 30.53 (Exceeds 25+ threshold for deployment)

**Calibration**:
- Scores properly distributed across 450-850 range
- Top 20% of borrowers score 550+
- Bottom 30% score below 520

**Feature Importance** (Top 5):
1. Interest Rate (`int_rate`) - 28.4%
2. Loan Grade (`grade`) - 19.7%
3. Debt-to-Income Ratio (`dti`) - 15.2%
4. Annual Income (`annual_inc`) - 12.8%
5. Revolving Utilization (`revol_util`) - 9.3%

---

## 🔍 Project Structure

```
credit-risk-engine/
│
├── app.py                          # Streamlit dashboard
├── train_model.py                  # Model training pipeline
├── lgb_credit_model.pkl            # Trained LightGBM model
├── woe_bins.pkl                    # WoE transformation rules
│
├── notebooks/
│   ├── credit-risk.ipynb            
│
├── data/
│   └── accepted_2007_to_2018Q4.csv.gz  # Raw dataset (not included)
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # MIT License
```

---

## 🐛 Troubleshooting

### Issue: "Model files not found"
**Solution**: Ensure `lgb_credit_model.pkl` and `woe_bins.pkl` exist in the project root. Run `train_model.py` if missing.

### Issue: "Ollama connection error"
**Solution**: 
```bash
# Check if Ollama is running
ollama list

# Start the server
ollama serve

# Verify Llama 3.2 is installed
ollama pull llama3.2
```

### Issue: "Memory error during training"
**Solution**: The Polars streaming pipeline should handle 2M+ rows with <1GB RAM. If issues persist:
```python
# Reduce chunk size in train_model.py
df = pl.scan_csv("data.csv").collect(streaming=True, slice=(0, 500000))
```

### Issue: "WoE transformation fails"
**Solution**: Ensure input data has the exact same column names as training data. Check for typos in feature names.

---

## 🔮 Future Enhancements

- [ ] **Model Monitoring**: MLflow integration for experiment tracking
- [ ] **Advanced Explainability**: SHAP values for feature importance
- [ ] **API Deployment**: FastAPI wrapper for production serving
- [ ] **A/B Testing**: Champion/challenger model framework
- [ ] **Real-time Updates**: Streaming data pipeline with Kafka
- [ ] **Database Integration**: PostgreSQL for application history
- [ ] **Enhanced UI**: React frontend with more visualizations
- [ ] **Model Registry**: Automated retraining pipelines

---

## 📚 Learning Resources

### Academic Papers
- [Credit Scoring Using Machine Learning](https://www.sciencedirect.com/science/article/pii/S0957417419306785)
- [Weight of Evidence in Credit Risk](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)

### Industry Standards
- [Basel III Capital Requirements](https://www.bis.org/bcbs/basel3.htm)
- [Fair Lending Regulations (ECOA)](https://www.consumerfinance.gov/rules-policy/regulations/1002/)

### Technical Documentation
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scorecardpy Package](https://github.com/ShichenXie/scorecardpy)
- [Polars User Guide](https://pola-rs.github.io/polars-book/)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Lending Club**: Dataset source (public domain)
- **Scorecardpy**: Excellent WoE/IV implementation
- **Ollama**: Local LLM infrastructure
- **Streamlit**: Rapid prototyping framework

---

## 📧 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/credit-risk-engine](https://github.com/yourusername/credit-risk-engine)

---

## ⭐ Star History

If this project helped you, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/credit-risk-engine&type=Date)](https://star-history.com/#yourusername/credit-risk-engine&Date)
