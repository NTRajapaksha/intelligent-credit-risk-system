import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import scorecardpy as sc
import ollama 

def generate_explanation(decision, probability, input_data):
    """
    Uses Llama 3.2 to generate a human-readable explanation.
    """
    # 1. Dynamically select the instruction based on the decision
    if "APPROVED" in decision:
        specific_instruction = "Congratulate them on the approval. Highlight their strong income or low debt as positive factors."
    else:
        specific_instruction = "Politely explain that the application was not instantly approved due to high risk factors (like Debt-to-Income ratio) and requires manual review."

    # 2. Build the focused prompt
    prompt = f"""
    You are a Senior Credit Risk Analyst at a bank.
    
    Task: Write a strict, professional 3-sentence explanation for this specific loan decision.
    
    Data:
    - Decision: {decision}
    - Risk Probability: {probability:.1%}
    - Applicant Income: ${input_data['annual_inc']}
    - Debt-to-Income: {input_data['dti']}
    
    Current Instruction: {specific_instruction}
    
    Do not mention model internals (like "LightGBM"). Speak like a banker. 
    """
    
    try:
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

# 1. Page Setup
st.set_page_config(page_title="Credit Risk Engine", layout="centered")
st.title("🏦 Intelligent Credit Scorecard")
st.markdown("---")

# 2. Load Assets (The MLOps Part)
@st.cache_resource
def load_assets():
    model = joblib.load('lgb_credit_model.pkl')
    with open('woe_bins.pkl', 'rb') as f:
        bins = pickle.load(f)
    return model, bins

try:
    model, bins = load_assets()
    st.success("Risk Model & Rules Engine Loaded Successfully")
except FileNotFoundError:
    st.error("Model files not found. Please run the training notebook first.")
    st.stop()

# 3. Sidebar Inputs (The Loan Application)
st.sidebar.header("📝 Applicant Details")

annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=10000, value=65000, step=1000)
dti = st.sidebar.slider("Debt-to-Income Ratio (DTI)", 0.0, 50.0, 15.0)
loan_amnt = st.sidebar.number_input("Loan Amount Requested ($)", min_value=1000, value=15000, step=500)
grade = st.sidebar.selectbox("Lending Club Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 12.5)
emp_length = st.sidebar.selectbox("Employment Length", 
    ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', 
     '6 years', '7 years', '8 years', '9 years', '10+ years'])

# Hidden defaults for features we need but didn't ask the user for (to keep UI simple)
# In a real app, these would come from a database or credit bureau API
default_input = {
    'term': ' 36 months',
    'installment': loan_amnt / 36, # Rough estimate
    'home_ownership': 'MORTGAGE',
    'verification_status': 'Source Verified',
    'purpose': 'debt_consolidation',
    'open_acc': 10,
    'pub_rec': 0,
    'revol_bal': 15000,
    'revol_util': 50,
    'total_acc': 25,
    'mort_acc': 1,
    'pub_rec_bankruptcies': 0,
    'credit_hist_years': 15 # Average
}

# 4. Processing Pipeline
if st.button("Calculate Risk Score"):
    
    # A. Create DataFrame from inputs
    input_data = default_input.copy()
    input_data.update({
        'annual_inc': annual_inc,
        'dti': dti,
        'loan_amnt': loan_amnt,
        'grade': grade,
        'int_rate': int_rate,
        'emp_length': emp_length
    })
    
    df_input = pd.DataFrame([input_data])
    
    # B. Apply WoE Transformation (The same rules as training!)
    # scorecardpy needs the columns to match exactly
    try:
        # We only transform the columns that exist in the bins
        # This handles the "Senior" logic of ensuring input consistency
        df_woe = sc.woebin_ply(df_input, bins)
    except Exception as e:
        st.error(f"Error in WoE Transformation: {e}")
        st.stop()
        
    # Reorder columns to match model training order
    # (LightGBM is sensitive to column order)
    model_cols = model.booster_.feature_name()
    
    # Ensure all columns exist, if not, fill with 0 (neutral WoE)
    for col in model_cols:
        if col not in df_woe.columns:
            df_woe[col] = 0.0
            
    df_woe = df_woe[model_cols]

    # C. Predict
    prob_default = model.predict_proba(df_woe)[:, 1][0]
    
    # D. Calibrate Score
    target_score = 600
    target_odds = 50
    pdo = 20
    factor = pdo / np.log(2)
    offset = target_score - (factor * np.log(target_odds))
    
    odds = (1 - prob_default) / (prob_default + 1e-10)
    score = offset + (factor * np.log(odds))
    score = int(np.clip(score, 300, 850))

    #5. Display Results
    st.subheader(f"Credit Score: {score}")
    
    # Updated Thresholds based on Model Distribution (Range: 450 - 570)
    # Top Tier (Green): Score > 550 (The top ~20% of borrowers)
    # Mid Tier (Yellow): Score 520 - 550
    # Bottom Tier (Red): Score < 520
    
    if score >= 550:
        color = "green"
        decision = "APPROVED"
    elif score >= 520:
        color = "orange"
        decision = "MANUAL REVIEW"
    else:
        color = "red"
        decision = "REJECTED"
        

    st.markdown(f"""
    <div style="background-color:{color};padding:20px;border-radius:10px;text-align:center;">
        <h2 style="color:white;margin:0;"> {decision}</h2>
        <p style="color:white;font-size:18px;">Probability of Default: {prob_default:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")
    st.subheader("🤖 AI Underwriter Analysis (Llama 3.2)")
    
    with st.spinner("Generating explanation..."):
        # Prepare data dict for the LLM
        llm_data = {
            'annual_inc': annual_inc,
            'dti': dti,
            'credit_hist_years': default_input['credit_hist_years']
        }
        
        explanation = generate_explanation(decision, prob_default, llm_data)
        st.write(explanation)
    
    # Feature Interpretation
    st.write("---")
    st.info("💡 **Why this score?** High Income and Grade 'A' increase score. High DTI reduces it.")
    
    # Add a "Debug" expander to show the raw values (impressive for technical demos)
    with st.expander("Show Technical Details"):
        st.write(f"**Baseline Score:** {offset:.0f}")
        st.write(f"**Odds:** {odds:.2f}")
        st.write(f"**Log Odds:** {np.log(odds):.2f}")