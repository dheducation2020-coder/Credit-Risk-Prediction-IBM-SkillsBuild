# AI-Driven Credit Risk Assessment & Loan Default Prediction
**AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship Program**  
*Conducted in partnership with BharatCares*

---

## Project Metadata
- **Student / Intern Name:** Soumojit Manna
- **Primary AI Development IDE:** IBM Bob
- **AI Analytics & Architecture Co-Pilot:** Gemini
- **Domain:** Banking & Financial Services (Credit Risk & Loan Underwriting)
- **Dataset:** [Loan Prediction Dataset 2025 (Kaggle)](https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025?resource=download) (20,000 records, 22 columns)
- **Target Variable:** `loan_paid_back` (1 = Paid Back / Low Risk, 0 = Default / High Risk)

---

## 1. Problem Statement & Financial Reality
Retail banks face an asymmetric loss structure when issuing consumer loans:
- **Missed Default (False Negative):** Approving a borrower who subsequently defaults results in catastrophic capital impairment (~$15,000 average write-off).
- **False Alarm (False Positive):** Flagging a creditworthy borrower costs only ~$50 for an underwriter to perform a manual document check.

This project delivers an end-to-end Machine Learning pipeline and an interactive Business Intelligence application prioritizing **Default Recall (73%)**, preserving **+$2,554,200.00 in loan principal** across a 4,000-loan test portfolio compared to standard accuracy-first models.

---

## 2. 5-Level Business Intelligence (BI) Decision Architecture
The application implements the 5-tier analytical hierarchy:
1. **Level 1 — Executive KPIs (What is happening?):** Portfolio Size (20,000 Loans), Baseline Default Rate (20.01%), Average Principal ($15,129.30), and Portfolio Credit Score (679 pts).
2. **Level 2 — Trajectories & Trends (Where is it going?):** Default rates stratified across bureau tiers, revealing sharp default jumps for credit scores below 580 (>30% default rate).
3. **Level 3 — Root Drivers (Why is it happening?):** Quantifying balance sheet leverage through Debt-to-Income (DTI) ratio distributions (40% higher median DTI for defaulters) and past delinquency counts.
4. **Level 4 — Operational Risk (What could go wrong?):** Asymmetric loss modeling demonstrating that high-accuracy models (e.g., Random Forest at 90%) can paradoxically incur over $5.7M in portfolio losses by failing to capture credit defaults.
5. **Level 5 — Prescriptive Actions (What should management do?):** Real-time underwriting rules engine with an interactive risk-gauge calculator prescribing three automated tiers:
   - *Default Risk < 30%:* Automated Approval (Prime Tier)
   - *Default Risk 30%–55%:* Route to Senior Underwriter (Manual Document Audit)
   - *Default Risk > 55%:* Adverse Action / Application Declined

---

## 3. Machine Learning Benchmark

| Metric | Logistic Regression (Class-Balanced) | Random Forest (Class-Balanced) | Selection Rationale |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 81% | 90% | Random Forest masks default misses |
| **Default Recall (Class 0)** | **73% (588/800 caught)** | 52% (416/800 caught) | **Logistic Regression prioritizes capital recovery** |
| **Default Precision (Class 0)** | 52% | 96% | False alarms cost only $50 review fee |
| **ROC-AUC Score** | **0.8857** | 0.8758 | Robust probability separation |
| **Portfolio Loss (4,000 Loans)** | **$3,206,650.00** | $5,760,850.00 | **+$2,554,200.00 Capital Preserved** |

---

## 4. Repository Structure
```text
├── loan_dataset_20000.csv                    # Clean borrower portfolio data
├── Soumojit_Manna_LoanDefaultPrediction.py   # Standalone Streamlit application
├── requirements.txt                          # Pinned dependency environment
├── README.md                                 # Project documentation
└── Soumojit_Manna_ProjectReport.docx         # Formal internship documentation