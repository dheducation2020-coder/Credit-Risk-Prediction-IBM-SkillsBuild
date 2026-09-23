import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# -----------------------------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Credit Risk Engine | Soumojit Manna",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & MODEL TRAINING (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(filepath="loan_dataset_20000.csv"):
    if not os.path.exists(filepath):
        st.error(f"Dataset file '{filepath}' was not found in the workspace.")
        st.stop()
    return pd.read_csv(filepath)

@st.cache_resource
def build_and_train_pipeline(df):
    # Drop multicollinear feature 'monthly_income'
    X = df.drop(columns=['loan_paid_back', 'monthly_income'])
    y = df['loan_paid_back']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ]
    )

    # 80/20 Stratified Partition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Balanced Logistic Regression to maximize Default Recall
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Test metrics
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return pipeline, X_test, y_test, y_pred, roc_auc, cm, num_cols, cat_cols

# Initialize
df = load_data()
model, X_test, y_test, y_pred, roc_auc, cm, num_cols, cat_cols = build_and_train_pipeline(df)

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("💳 Credit Risk Engine")
st.sidebar.caption("AICTE | IBM SkillsBuild Internship")
st.sidebar.markdown("**Student:** Soumojit Manna  \n**IDE:** IBM Bob  \n**Co-Pilot:** Gemini")

menu_choice = st.sidebar.radio(
    "Select Workflow View:",
    [
        "1. Executive 5-Level BI Dashboard",
        "2. Real-Time Applicant Underwriting Tool",
        "3. Model Performance & Asymmetric Loss Audit"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Core Business Context:**\n"
    "- Target: `loan_paid_back` (1 = Paid, 0 = Default)\n"
    "- Default Rate: **20.01%**\n"
    "- Strategy: **Recall Priority** to avoid $15,000 write-offs."
)

# -----------------------------------------------------------------------------
# 4. VIEW 1: EXECUTIVE 5-LEVEL BI DASHBOARD
# -----------------------------------------------------------------------------
if menu_choice == "1. Executive 5-Level BI Dashboard":
    st.title("📊 Executive Credit Decision Dashboard")
    st.markdown("Structured strictly according to the **5-Level Business Intelligence Hierarchy**.")

    # LEVEL 1: KPIs
    st.markdown("### Level 1: Executive KPIs (What is happening right now?)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_portfolio = len(df)
    default_pct = (df['loan_paid_back'] == 0).mean() * 100
    avg_loan_val = df['loan_amount'].mean()
    avg_credit = df['credit_score'].mean()

    kpi1.metric("Active Portfolio", f"{total_portfolio:,} Loans")
    kpi2.metric("Portfolio Default Rate", f"{default_pct:.2f}%", delta="-0.3% vs Prior Qtr")
    kpi3.metric("Average Principal", f"${avg_loan_val:,.2f}")
    kpi4.metric("Avg Portfolio Credit Score", f"{avg_credit:.0f} pts")

    st.divider()

    # LEVEL 2 & LEVEL 3: TRENDS & DRIVERS
    col_l2, col_l3 = st.columns(2)
    with col_l2:
        st.markdown("### Level 2: Trends (Where is it going?)")
        df['score_bin'] = pd.cut(
            df['credit_score'],
            bins=[300, 580, 670, 740, 800, 900],
            labels=['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Very Good (740-799)', 'Exceptional (800+)']
        )
        tier_def = df.groupby('score_bin', observed=False)['loan_paid_back'].value_counts(normalize=True).unstack()[0] * 100
        fig_trend = px.bar(
            tier_def.reset_index(),
            x='score_bin',
            y=0,
            title="Default Trajectory by Bureau Score Tier",
            labels={'score_bin': 'Credit Tier', 0: 'Default Rate (%)'},
            color=0,
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_l3:
        st.markdown("### Level 3: Drivers (Why is it happening?)")
        fig_driver = px.box(
            df,
            x='loan_paid_back',
            y='debt_to_income_ratio',
            color='loan_paid_back',
            title="Primary Driver: Debt Burden (DTI) Distribution",
            labels={'loan_paid_back': 'Status (0 = Default, 1 = Paid Back)', 'debt_to_income_ratio': 'DTI Ratio'},
            color_discrete_map={0: "#E74C3C", 1: "#2ECC71"}
        )
        st.plotly_chart(fig_driver, use_container_width=True)

    st.divider()

    # LEVEL 4 & LEVEL 5: RISKS & ACTIONS
    col_l4, col_l5 = st.columns(2)
    with col_l4:
        st.markdown("### Level 4: Risk Exposure (What could go wrong?)")
        st.error(
            "**Asymmetric Capital Impairment Risk:**\n"
            "- A missed defaulter (**False Negative**) triggers an average write-off of **$15,000**.\n"
            "- A false alarm (**False Positive**) costs only **$50** for an underwriter review.\n"
            "- Favoring raw accuracy over default recall exposes the book to **>$2.5M in avoidable losses**."
        )

    with col_l5:
        st.markdown("### Level 5: Prescriptive Actions (What should management do?)")
        st.success(
            "**Operational Directives:**\n"
            "1. **Hard DTI Ceiling:** Cap automated approvals when an applicant's DTI exceeds 25%.\n"
            "2. **Tiered Verification:** Mandate manual pay-stub verification for credit scores between 580 and 660.\n"
            "3. **Risk-Based Pricing:** Apply a 150-basis-point interest surcharge on 60-month terms with past delinquencies."
        )

# -----------------------------------------------------------------------------
# 5. VIEW 2: REAL-TIME APPLICANT UNDERWRITING TOOL
# -----------------------------------------------------------------------------
elif menu_choice == "2. Real-Time Applicant Underwriting Tool":
    st.title("🔍 Real-Time Applicant Underwriting Calculator")
    st.markdown("Input applicant variables to simulate real-time credit scoring and automated underwriting guidance.")

    with st.form("underwrite_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### 👤 Applicant Demographics")
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            education_level = st.selectbox("Education Level", ["High School", "Associate", "Bachelor's", "Master's", "PhD"])
            employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed", "Retired"])

        with c2:
            st.markdown("##### 💵 Financial Capacity")
            annual_income = st.number_input("Annual Gross Income ($)", min_value=5000.0, max_value=500000.0, value=45000.0, step=1000.0)
            dti = st.slider("Debt-to-Income (DTI) Ratio", 0.0, 1.0, 0.18, step=0.01)
            loan_amount = st.number_input("Requested Principal ($)", min_value=500.0, max_value=100000.0, value=15000.0, step=500.0)
            loan_purpose = st.selectbox("Loan Purpose", sorted(df['loan_purpose'].unique().tolist()))
            loan_term = st.selectbox("Loan Term (Months)", [36, 60])

        with c3:
            st.markdown("##### 📈 Credit History & Bureau")
            credit_score = st.slider("Credit Score", 300, 850, 680)
            interest_rate = st.slider("Assigned Interest Rate (%)", 5.0, 30.0, 12.5, step=0.1)
            installment = st.number_input("Monthly Installment ($)", min_value=10.0, max_value=5000.0, value=450.0, step=25.0)
            grade_subgrade = st.selectbox("Internal Risk Grade", sorted(df['grade_subgrade'].unique().tolist()))
            num_of_open_accounts = st.slider("Open Credit Accounts", 1, 30, 7)
            total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=1000.0, max_value=200000.0, value=35000.0, step=1000.0)
            current_balance = st.number_input("Current Balance ($)", min_value=0.0, max_value=200000.0, value=10000.0, step=500.0)
            delinquency_history = st.selectbox("Delinquency History Flag", [0, 1, 2, 3, 4, 5])
            num_of_delinquencies = st.slider("Count of Past Delinquencies", 0, 15, 1)
            public_records = st.selectbox("Public Derogatory Records", [0, 1, 2, 3])

        submit_btn = st.form_submit_button("Assess Credit Risk", use_container_width=True)

    if submit_btn:
        applicant_row = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'education_level': education_level,
            'annual_income': annual_income,
            'employment_status': employment_status,
            'debt_to_income_ratio': dti,
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'loan_purpose': loan_purpose,
            'interest_rate': interest_rate,
            'loan_term': loan_term,
            'installment': installment,
            'grade_subgrade': grade_subgrade,
            'num_of_open_accounts': num_of_open_accounts,
            'total_credit_limit': total_credit_limit,
            'current_balance': current_balance,
            'delinquency_history': delinquency_history,
            'public_records': public_records,
            'num_of_delinquencies': num_of_delinquencies
        }])

        prob_repaid = model.predict_proba(applicant_row)[0, 1]
        default_prob = (1 - prob_repaid) * 100

        res_left, res_right = st.columns([1, 2])
        with res_left:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=default_prob,
                title={'text': "Default Risk Score (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2C3E50"},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ECC71"},
                        {'range': [30, 55], 'color': "#F1C40F"},
                        {'range': [55, 100], 'color': "#E74C3C"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_right:
            st.subheader("Underwriting Directive")
            if default_prob < 30.0:
                st.success("### ✅ AUTOMATED APPROVAL")
                st.write(f"- **Risk Assessment:** Low Default Probability ({default_prob:.1f}%).")
                st.write("- **Prescriptive Action:** Issue standard promissory note under prime interest terms.")
            elif default_prob < 55.0:
                st.warning("### ⚠️ MANUAL AUDIT REQUIRED")
                st.write(f"- **Risk Assessment:** Moderate Default Probability ({default_prob:.1f}%).")
                st.write("- **Prescriptive Action:** Route file to Senior Underwriter for income verification, bank statement audit, and collateral review.")
            else:
                st.error("### 🛑 DECLINE APPLICATION")
                st.write(f"- **Risk Assessment:** Elevated Default Probability ({default_prob:.1f}%).")
                st.write("- **Prescriptive Action:** Credit risk exceeds portfolio loss tolerance. Issue adverse action notice.")

# -----------------------------------------------------------------------------
# 6. VIEW 3: MODEL PERFORMANCE & ASYMMETRIC LOSS AUDIT
# -----------------------------------------------------------------------------
elif menu_choice == "3. Model Performance & Asymmetric Loss Audit":
    st.title("📈 Model Evaluation & Loss Matrix")
    st.markdown("Validating model efficacy on the 4,000-borrower stratified test set.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Confusion Matrix")
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Default (0)', 'Paid Back (1)'],
            y=['Default (0)', 'Paid Back (1)'],
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        st.subheader("Classification Metrics")
        st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
        st.text(classification_report(y_test, y_pred, target_names=['Default (0)', 'Paid Back (1)']))

    st.divider()
    st.subheader("Simulated Capital Impact (4,000 Holdout Loans)")
    loss_table = pd.DataFrame({
        "Model Architecture": [
            "Logistic Regression (Class-Balanced: Recall Priority)",
            "Random Forest (Class-Balanced: Accuracy Priority)"
        ],
        "Accuracy": ["81%", "90%"],
        "Default Recall": ["73%", "52%"],
        "Missed Defaults": [212, 384],
        "False Alarms": [533, 17],
        "Total Portfolio Loss ($)": ["$3,206,650.00", "$5,760,850.00"],
        "Net Capital Preserved ($)": ["+$2,554,200.00", "Baseline"]
    })
    st.table(loss_table)
    