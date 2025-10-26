import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin

# ========== CUSTOM CLASSES (Required for model loading) ==========
class NoOutlier(BaseEstimator, TransformerMixin):
    """Custom transformer for outlier handling"""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X, y=None):
        return X

class CustomImputer(BaseEstimator, TransformerMixin):
    """Custom imputer for totalcharges column"""
    def __init__(self, median_value=1397.475):
        self.median_value = median_value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            X['TotalCharges'].fillna(self.median_value, inplace=True)
        return X

# Custom function (for backward compatibility)
def impute_totalcharges(X):
    """Custom imputer function for totalcharges"""
    X = X.copy()
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        median_value = 1397.475
        X['TotalCharges'].fillna(median_value, inplace=True)
    return X

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .churn-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .churn-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL & DATA ==========
@st.cache_resource
def load_model():
    return joblib.load('churn_model.joblib')

@st.cache_data
def load_train_data():
    df = pd.read_csv('customerchurn.csv')
    # Drop customerID dan Churn (kolom yang tidak dipakai untuk training)
    columns_to_drop = ['customerID', 'Churn']
    X_train = df.drop(columns_to_drop, axis=1, errors='ignore')
    return X_train

try:
    model = load_model()
    X_train = load_train_data()
    model_loaded = True
    
    # Debug info di sidebar
    st.sidebar.success(f"✅ Model loaded successfully!")
    st.sidebar.info(f"Features: {len(X_train.columns)}")
    
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Error loading model: {e}")
    st.info("Please make sure 'churn_model.joblib' and 'customerchurn.csv' are in the same directory as app.py")

# ========== HEADER ==========
st.markdown('<p class="main-header">📊 Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict customer churn probability using Machine Learning</p>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## 📋 About")
    st.info("""
    This app predicts customer churn probability using:
    - **Model**: LogisticRegression
    - **Preprocessing**: RobustScaler + SMOTEENN
    - **Features**: 19 customer attributes
    - **Accuracy**: 77.1%
    - **Recall**: 80.0%
    """)
    
    st.markdown("## 🎯 How to Use")
    st.markdown("""
    1. Fill in customer information
    2. Click **Predict Churn**
    3. View prediction results
    4. Check recommendations
    """)

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Dataset Insights"])

# ========== TAB 1: PREDICTION ==========
with tab1:
    st.markdown("### 👤 Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📝 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender_select")
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="senior_select")
        partner = st.selectbox("Partner", ["Yes", "No"], key="partner_select")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents_select")
    
    with col2:
        st.markdown("#### 📞 Services")
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"], key="phone_select")
        multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="lines_select")
        internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_select")
        onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="security_select")
        onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup_select")
    
    with col3:
        st.markdown("#### 🛡️ Protection & Support")
        deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device_select")
        techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech_select")
        streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="tv_select")
        streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="movie_select")
    
    st.markdown("---")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("#### 📄 Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract_select")
        paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"], key="billing_select")
        paymentmethod = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="payment_select")
    
    with col5:
        st.markdown("#### 💰 Charges")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure_slider")
        monthlycharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 0.1, key="monthly_input")
    
    with col6:
        st.markdown("#### 💳 Total")
        totalcharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 
                                       float(tenure * monthlycharges), 0.1, key="total_input")
    
    st.markdown("---")
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Churn", key="predict_btn", use_container_width=True)
    
    if predict_button and model_loaded:
        try:
            # Buat data dictionary dengan nama kolom PERSIS seperti dataset asli
            input_data = {
                'gender': [str(gender)],
                'SeniorCitizen': [int(senior)],
                'Partner': [str(partner)],
                'Dependents': [str(dependents)],
                'tenure': [int(tenure)],
                'PhoneService': [str(phoneservice)],
                'MultipleLines': [str(multiplelines)],
                'InternetService': [str(internetservice)],
                'OnlineSecurity': [str(onlinesecurity)],
                'OnlineBackup': [str(onlinebackup)],
                'DeviceProtection': [str(deviceprotection)],
                'TechSupport': [str(techsupport)],
                'StreamingTV': [str(streamingtv)],
                'StreamingMovies': [str(streamingmovies)],
                'Contract': [str(contract)],
                'PaperlessBilling': [str(paperlessbilling)],
                'PaymentMethod': [str(paymentmethod)],
                'MonthlyCharges': [float(monthlycharges)],
                'TotalCharges': [float(totalcharges)]
            }
            
            # Buat DataFrame
            dummy = pd.DataFrame(input_data)
            
            # Pastikan urutan kolom sesuai X_train
            dummy = dummy[X_train.columns]
            
            # Predict
            prediction = model.predict(dummy)[0]
            probability = model.predict_proba(dummy)[0]
            
            churn_prob = probability[1]
            no_churn_prob = probability[0]
            
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {str(e)}")
            st.write("**Debug Info:**")
            st.write("Expected columns:", X_train.columns.tolist())
            st.write("Input columns:", list(input_data.keys()))
            import traceback
            st.code(traceback.format_exc())
            st.stop()
        
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Result cards
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            if prediction == 1:
                st.markdown(f"""
                <div class="metric-card churn-high">
                    <h2>⚠️ HIGH RISK</h2>
                    <h1>CHURN</h1>
                    <p>Customer likely to leave</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card churn-low">
                    <h2>✅ LOW RISK</h2>
                    <h1>NO CHURN</h1>
                    <p>Customer likely to stay</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_r2:
            st.metric("Churn Probability", f"{churn_prob:.1%}", 
                     delta=f"{churn_prob-0.5:.1%}",
                     delta_color="inverse")
        
        with col_r3:
            st.metric("Retention Probability", f"{no_churn_prob:.1%}",
                     delta=f"{no_churn_prob-0.5:.1%}")
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk Level", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': '#00f2fe'},
                    {'range': [30, 70], 'color': '#ffd89b'},
                    {'range': [70, 100], 'color': '#f5576c'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("## 💡 Recommendations")
        
        if prediction == 1:
            st.error("### ⚠️ HIGH CHURN RISK - Immediate Action Required!")
            
            recommendations = []
            
            if contract == "Month-to-month":
                recommendations.append("🎯 **Contract Upgrade**: Offer incentive for 1-year or 2-year contract")
            
            if tenure < 12:
                recommendations.append("👋 **New Customer**: Enhance onboarding experience and early engagement")
            
            if internetservice == "Fiber optic":
                recommendations.append("🌐 **Fiber Optic User**: Review service quality and pricing competitiveness")
            
            if paymentmethod == "Electronic check":
                recommendations.append("💳 **Payment Method**: Encourage automatic payment methods")
            
            if techsupport == "No" or techsupport == "No internet service":
                recommendations.append("🛠️ **Tech Support**: Offer complimentary tech support trial")
            
            if onlinesecurity == "No" or onlinesecurity == "No internet service":
                recommendations.append("🔒 **Security Services**: Bundle online security at discounted rate")
            
            if monthlycharges > 70:
                recommendations.append("💰 **High Charges**: Review pricing plan and offer loyalty discounts")
            
            if not recommendations:
                recommendations.append("📞 **Proactive Outreach**: Contact customer to understand concerns")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
        else:
            st.success("### ✅ LOW CHURN RISK - Maintain & Grow!")
            st.markdown("""
            - 🎁 **Loyalty Rewards**: Enroll in loyalty program for continued engagement
            - 📢 **Upsell Opportunities**: Introduce premium services or add-ons
            - 🌟 **Referral Program**: Encourage customer referrals with incentives
            - 📊 **Feedback Loop**: Regular satisfaction surveys to maintain quality
            """)

# ========== TAB 2: MODEL PERFORMANCE ==========
with tab2:
    st.markdown("## 📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "77.1%")
    with col2:
        st.metric("Recall", "80.0%")
    with col3:
        st.metric("Precision", "56.4%")
    with col4:
        st.metric("F2 Score", "0.738")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Confusion Matrix")
        
        cm_data = [[770, 244], [79, 316]]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Predicted No Churn', 'Predicted Churn'],
            y=['Actual No Churn', 'Actual Churn'],
            colorscale='Blues',
            text=cm_data,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col_right:
        st.markdown("### Key Insights")
        st.info("""
        **Model correctly identified:**
        - ✅ 770 loyal customers (True Negative)
        - ✅ 316 churners (True Positive)
        
        **Model mistakes:**
        - ⚠️ 244 false alarms (False Positive)
        - ❌ 79 missed churners (False Negative)
        
        **Business Impact:**
        - 80% of actual churners detected
        - 20% of churners might be missed
        - Cost-effective for retention programs
        """)

# ========== TAB 3: DATASET INSIGHTS ==========
with tab3:
    st.markdown("## 📈 Dataset Insights")
    
    st.markdown("### Feature Importance (Top 10)")
    
    feature_importance = pd.DataFrame({
        'Feature': ['tenure', 'paperlessbilling', 'totalcharges', 'contract_Two year', 
                   'onlinebackup', 'internetservice', 'dependents', 'phoneservice',
                   'multiplelines', 'monthlycharges'],
        'Importance': [3.44, 2.08, 1.76, 1.01, 0.65, 0.63, 0.58, 0.60, 0.34, 0.29]
    })
    
    fig_importance = px.bar(feature_importance, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Viridis')
    fig_importance.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("### Key Findings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Top Retention Factors:**
        - 📅 **Long Tenure**: Customers staying longer are more loyal
        - 📄 **Two-Year Contract**: Strong commitment indicator
        - 💾 **Online Backup**: Service adoption shows engagement
        """)
    
    with col2:
        st.error("""
        **Top Churn Indicators:**
        - 📝 **Paperless Billing**: Digital engagement matters
        - 💰 **Total Charges**: High costs need value justification
        - 🌐 **Internet Service Type**: Fiber optic users churn more
        """)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Customer Churn Prediction System | Powered by Machine Learning</p>
    <p>Model Accuracy: 77.1% | Recall: 80.0% | F2 Score: 0.738</p>
</div>
""", unsafe_allow_html=True)
