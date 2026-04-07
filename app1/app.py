import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

# Force background plotting to prevent multi-threading conflicts in web apps
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 0. Page Configuration & CSS Styling
# ==========================================
st.set_page_config(
    page_title="Nutritional-Inflammatory Imbalance Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# 💉 Custom Advanced CSS
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Times New Roman', sans-serif;
    }
    
    /* Core button styling */
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 15px;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B4F72;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F8F9F9;
        border-right: 1px solid #E5E7E9;
    }
    
    /* Metric value color reinforcement */
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        color: #C0392B; 
        font-weight: 900;
    }
    
    /* Input field styling */
    input[type="number"] {
        font-weight: bold;
        color: #154360;
        background-color: #F4F6F7;
    }
    
    /* Hide native menu and deploy button */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Header Design
# ==========================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) 
with col_title:
    st.title("Intelligent Warning Platform for Postoperative Nutritional-Inflammatory Imbalance")
    st.markdown("**(Clinical Decision Support System based on Artificial Neural Network)**")

st.markdown("""
<div style='background-color: #EBF5FB; padding: 15px; border-radius: 10px; border-left: 5px solid #2980B9; margin-bottom: 25px;'>
    <span style='color: #154360; font-size: 15px;'>
    <b>📊 System Introduction:</b> Powered by an advanced <b>Artificial Neural Network (ANN)</b> algorithm, this platform integrates 6 core preoperative clinical indicators to <b>dynamically predict the risk of postoperative nutritional-inflammatory imbalance (indicated by AGR inversion)</b> in colon cancer patients. It features multi-view <b>SHAP (SHapley Additive exPlanations)</b> for individualized interpretation, providing clinicians with intuitive and explainable decision support.
    </span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. ANN Model & Scaler Loading Engine (Cloud Version)
# ==========================================
@st.cache_resource 
def load_assets():
    # 动态获取当前脚本所在的目录，完美兼容 Github 云端路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "ann_model.pkl")
    scaler_path = os.path.join(base_dir, "ann_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Missing model or scaler file in the repository. Please ensure 'ann_model.pkl' and 'ann_scaler.pkl' are in the same folder as app.py.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
            
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("🚨 Loading failed. Please check your GitHub repository files.")
    st.warning(f"System Error: {e}")
    st.stop()


# ==========================================
# 3. Two-way Binding for Inputs
# ==========================================
default_values = {
    'WBC': 6.92, 'ALB': 42.5, 'GLO': 27.0, 
    'PA': 235.2, 'ChE': 8655.0, 'FDP': 0.22
}

for key, val in default_values.items():
    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = val
    if f"{key}_num" not in st.session_state:
        st.session_state[f"{key}_num"] = val

def sync_inputs(src_key, dest_key):
    st.session_state[dest_key] = st.session_state[src_key]


# ==========================================
# 4. Sidebar: Quick Sliders 
# ==========================================
st.sidebar.markdown("### 🖥️ System Status")
st.sidebar.success("🟢 Core Engine: ANN Ready")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎛️ Rapid Parameter Adjustment")

with st.sidebar.expander("🩸 Core Hematological Indices", expanded=True):
    st.slider("White Blood Cell (WBC) ×10^9/L", 0.0, 50.0, step=0.01, key="WBC_slider", on_change=sync_inputs, args=("WBC_slider", "WBC_num"))
    st.slider("Albumin (ALB) g/L", 10.0, 80.0, step=0.1, key="ALB_slider", on_change=sync_inputs, args=("ALB_slider", "ALB_num"))
    st.slider("Globulin (GLO) g/L", 10.0, 80.0, step=0.1, key="GLO_slider", on_change=sync_inputs, args=("GLO_slider", "GLO_num"))

with st.sidebar.expander("🔬 Specific Proteins & Enzymes", expanded=True):
    st.slider("Prealbumin (PA) mg/L", 10.0, 600.0, step=1.0, key="PA_slider", on_change=sync_inputs, args=("PA_slider", "PA_num"))
    st.slider("Cholinesterase (ChE) U/L", 100.0, 15000.0, step=10.0, key="ChE_slider", on_change=sync_inputs, args=("ChE_slider", "ChE_num"))
    st.slider("Fibrin Degradation Products (FDP) mg/L", 0.0, 120.0, step=0.01, key="FDP_slider", on_change=sync_inputs, args=("FDP_slider", "FDP_num"))


# ==========================================
# 5. Main Content: Precise Input Matrix
# ==========================================
st.markdown("### 👨‍⚕️ Clinical Parameter Input Matrix")
st.markdown("*(Enter exact values below, or use the sidebar sliders to adjust synchronously)*")

col1, col2, col3 = st.columns(3)
with col1:
    st.number_input("WBC (×10^9/L)", min_value=0.0, max_value=50.0, step=0.01, format="%.2f", key="WBC_num", on_change=sync_inputs, args=("WBC_num", "WBC_slider"))
    st.number_input("PA (mg/L)", min_value=10.0, max_value=600.0, step=1.0, format="%.1f", key="PA_num", on_change=sync_inputs, args=("PA_num", "PA_slider"))
with col2:
    st.number_input("ALB (g/L)", min_value=10.0, max_value=80.0, step=0.1, format="%.1f", key="ALB_num", on_change=sync_inputs, args=("ALB_num", "ALB_slider"))
    st.number_input("ChE (U/L)", min_value=100.0, max_value=15000.0, step=10.0, format="%.0f", key="ChE_num", on_change=sync_inputs, args=("ChE_num", "ChE_slider"))
with col3:
    st.number_input("GLO (g/L)", min_value=10.0, max_value=80.0, step=0.1, format="%.1f", key="GLO_num", on_change=sync_inputs, args=("GLO_num", "GLO_slider"))
    st.number_input("FDP (mg/L)", min_value=0.0, max_value=120.0, step=0.01, format="%.2f", key="FDP_num", on_change=sync_inputs, args=("FDP_num", "FDP_slider"))

# Construct Feature DataFrame (Strict order for ANN)
input_df = pd.DataFrame({
    'WBC': [st.session_state["WBC_num"]], 
    'ALB': [st.session_state["ALB_num"]], 
    'GLO': [st.session_state["GLO_num"]], 
    'PA': [st.session_state["PA_num"]],
    'ChE': [st.session_state["ChE_num"]],
    'FDP': [st.session_state["FDP_num"]]
})

# ==========================================
# 6. Core Prediction & Explanation Engine
# ==========================================
if st.button("🚀 Run AI Risk Assessment", type="primary"):
    with st.spinner('🧬 ANN is calculating forward propagation and multi-view explanations...'):
        
        # 1. Data Normalization for ANN
        try:
            if scaler is not None:
                scaled_input = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
            else:
                scaled_input = input_df
        except Exception as e:
            st.error("Data normalization failed. Please ensure feature names match training data.")
            st.stop()
        
        # 2. Risk probability prediction
        risk_prob = model.predict_proba(scaled_input)[0][1] 
        
        st.markdown("---")
        st.markdown("### 🎯 Postoperative Risk Inference Report")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Probability of Imbalance", value=f"{risk_prob * 100:.2f} %")
            
        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True) 
            if risk_prob > 0.5: 
                st.error("🚨 **[HIGH RISK ALERT]** The model identifies this patient as highly susceptible to severe **postoperative nutritional-inflammatory imbalance (indicated by A/G < 1)**. This highlights a critical depletion of nutritional reserves coupled with profound systemic inflammatory stress. Enhanced perioperative monitoring and proactive intervention are strongly recommended.")
                st.toast('High-risk alert detected!', icon='⚠️') 
            else:
                st.success("✅ **[SAFE ASSESSMENT]** The patient is currently in the low-risk zone. No significant tendency for severe postoperative nutritional-inflammatory imbalance detected. Maintenance of standard postoperative care protocols is recommended.")
                st.balloons() 

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 Risk Factor Attribution (Multiple Interpretability Views)")
        st.info("💡 **Interpretation Guide:** Explore different tabs to view the SHAP explanations from multiple perspectives. Red color indicates risk-increasing factors, while blue indicates protective factors.")
        
        try:
            # 3. KernelExplainer setup for ANN
            background = np.zeros((1, scaled_input.shape[1]))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values_raw = explainer.shap_values(scaled_input)
            
            # 4. Smart Shape Diagnosis (Extracting positive class)
            if isinstance(shap_values_raw, list):
                shap_val_single = shap_values_raw[1][0]
                base_val = explainer.expected_value[1]
            else:
                shap_array = np.array(shap_values_raw)
                if len(shap_array.shape) == 3:
                    shap_val_single = shap_array[0, :, 1]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                elif len(shap_array.shape) == 2 and shap_array.shape[1] == 2 and scaled_input.shape[1] == 6:
                    shap_val_single = shap_array[:, 1]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    shap_val_single = shap_array[0]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[0]
                
            # Create Explanation object (mapped back to raw unscaled input_df for clinical readability)
            exp = shap.Explanation(values=shap_val_single, base_values=base_val, 
                                   data=input_df.iloc[0], feature_names=input_df.columns.tolist())
            
            # 5. Render Multi-Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["🌊 Waterfall Plot", "⚖️ Force Plot", "📈 Decision Plot", "📊 Bar Plot"])
            
            with tab1:
                st.markdown("#### 1. Local Waterfall Plot")
                st.write("Details how each feature contributes incrementally, starting from the baseline value to reach the final prediction.")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(exp, max_display=10, show=False)
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
                
            with tab2:
                st.markdown("#### 2. Local Force Plot")
                st.write("Visualizes the competing forces of clinical features. Red features push the risk higher, while blue features push it lower.")
                shap.force_plot(base_val, shap_val_single, input_df.iloc[0], matplotlib=True, show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()
                
            with tab3:
                st.markdown("#### 3. Decision Plot")
                st.write("Traces the cumulative effect of features along a decision path, illustrating how the final clinical decision is shaped.")
                shap.decision_plot(base_val, shap_val_single, input_df.iloc[0], show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()
                
            with tab4:
                st.markdown("#### 4. Absolute Impact Bar Plot")
                st.write("Ranks the patient's individual features strictly by their absolute impact magnitude on the current prediction.")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(exp, max_display=10, show=False)
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
            
        except Exception as e:
            st.error(f"An error occurred while generating the SHAP plots: {e}")