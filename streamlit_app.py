import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
st.set_page_config(page_title="CardioGuard XAI", layout="wide", page_icon="🫀")

# --- Title and Description ---
st.title("🫀 CardioGuard XAI: Heart Disease Dashboard")
st.markdown("""
Welcome to CardioGuard XAI. Use the sidebar to simulate patient data and see real-time predictions. 
Navigate through the tabs below to explore individual predictions, global model behaviors, performance metrics, and the raw dataset.
""")

# --- Caching Data Loading and Model Training ---
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "heart.csv")
    df = pd.read_csv(file_path)
    return df

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Train/Test Split for evaluation metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_explainer = shap.TreeExplainer(rf_model)
    
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    eval_metrics = {
        'Random Forest': {
            'Accuracy': accuracy_score(y_test, rf_preds),
            'Precision': precision_score(y_test, rf_preds),
            'Recall': recall_score(y_test, rf_preds),
            'F1': f1_score(y_test, rf_preds),
            'ROC-AUC': roc_auc_score(y_test, rf_probs),
            'Confusion Matrix': confusion_matrix(y_test, rf_preds),
            'FPR': roc_curve(y_test, rf_probs)[0],
            'TPR': roc_curve(y_test, rf_probs)[1]
        }
    }
    
    models_dict = {'Random Forest': (rf_model, rf_explainer)}
    
    if HAS_XGB:
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_explainer = shap.TreeExplainer(xgb_model)
        
        xgb_preds = xgb_model.predict(X_test)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        
        eval_metrics['XGBoost'] = {
            'Accuracy': accuracy_score(y_test, xgb_preds),
            'Precision': precision_score(y_test, xgb_preds),
            'Recall': recall_score(y_test, xgb_preds),
            'F1': f1_score(y_test, xgb_preds),
            'ROC-AUC': roc_auc_score(y_test, xgb_probs),
            'Confusion Matrix': confusion_matrix(y_test, xgb_preds),
            'FPR': roc_curve(y_test, xgb_probs)[0],
            'TPR': roc_curve(y_test, xgb_probs)[1]
        }
        models_dict['XGBoost'] = (xgb_model, xgb_explainer)
    
    return models_dict, eval_metrics, X, y

# Load data and models
with st.spinner("Initializing models and computing explanations..."):
    df = load_data()
    models, eval_metrics, X_full, y_full = train_models(df)

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Configuration")

model_options = ["Random Forest"]
if HAS_XGB:
    model_options.append("XGBoost")
else:
    st.sidebar.warning("⚠️ XGBoost disabled: missing `libomp` (macOS system dependency).")
    
model_choice = st.sidebar.selectbox("Select Prediction Model", model_options)
model, explainer = models[model_choice]

st.sidebar.markdown("---")
st.sidebar.header("👤 Patient Profile")

with st.sidebar.expander("Demographics", expanded=True):
    age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

with st.sidebar.expander("Vitals & Symptoms", expanded=True):
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: f"Type {x}")
    trestbps = st.slider("Resting Blood Pressure", int(df['trestbps'].min()), int(df['trestbps'].max()), int(df['trestbps'].mean()))
    thalach = st.slider("Maximum Heart Rate Achieved", int(df['thalach'].min()), int(df['thalach'].max()), int(df['thalach'].mean()))
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with st.sidebar.expander("Lab Results & ECG", expanded=True):
    chol = st.slider("Cholesterol (mg/dl)", int(df['chol'].min()), int(df['chol'].max()), int(df['chol'].mean()))
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False")
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    oldpeak = st.slider("ST Depression (Oldpeak)", float(df['oldpeak'].min()), float(df['oldpeak'].max()), float(df['oldpeak'].mean()), 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.slider("Number of Major Vessels", 0, 4, int(df['ca'].mode()[0]))
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

user_data = {
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}
input_df = pd.DataFrame(user_data, index=[0])

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Local Prediction (Patient)", 
    "🌍 Global Insights (Model)", 
    "📊 Model Performance",
    "📈 Data Explorer"
])

# --- TAB 1: Local Prediction ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    with col1:
        st.subheader("Model Diagnosis")
        if prediction == 1:
            st.error("🚨 **High Risk of Heart Disease**")
        else:
            st.success("✅ **Low Risk of Heart Disease**")
            
        st.markdown("### Risk Probability")
        pcol1, pcol2 = st.columns(2)
        pcol1.metric("Low Risk (0)", f"{prediction_proba[0]:.1%}")
        pcol2.metric("High Risk (1)", f"{prediction_proba[1]:.1%}")

    with col2:
        st.subheader(f"Why? ({model_choice} SHAP Explanation)")
        st.write("This waterfall plot illustrates how each feature shifted the model's prediction from the baseline risk.")
        
        shap_values = explainer(input_df)
        if len(shap_values.shape) == 3: # Multi-class like RF
            shap_values_class1 = shap_values[..., 1]
        else: # Binary like XGBoost default
            shap_values_class1 = shap_values
            
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.plots.waterfall(shap_values_class1[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# --- TAB 2: Global Insights ---
with tab2:
    st.subheader(f"Global Feature Importance ({model_choice})")
    st.write("This plot shows the overall impact of each feature across all patients in the dataset. Features at the top are the most important drivers for the model's predictions.")
    
    # Calculate SHAP values for a sample of the data (for speed)
    @st.cache_resource
    def get_global_shap_values(model_name):
        # We reuse the trained explainer
        sample_X = X_full.sample(min(len(X_full), 200), random_state=42)
        global_shap_values = explainer(sample_X)
        if len(global_shap_values.shape) == 3:
            return global_shap_values[..., 1]
        return global_shap_values
        
    g_shap = get_global_shap_values(model_choice)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.summary_plot(g_shap, features=X_full.sample(min(len(X_full), 200), random_state=42), show=False)
    st.pyplot(fig2)
    plt.close(fig2)

# --- TAB 3: Model Performance ---
with tab3:
    st.subheader(f"{model_choice} Performance Metrics")
    metrics = eval_metrics[model_choice]
    
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    mcol1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    mcol2.metric("Precision", f"{metrics['Precision']:.2%}")
    mcol3.metric("Recall", f"{metrics['Recall']:.2%}")
    mcol4.metric("F1-Score", f"{metrics['F1']:.2%}")
    mcol5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2%}")
    
    st.markdown("---")
    col_cm, col_roc = st.columns(2)
    
    with col_cm:
        st.write("**Confusion Matrix (Test Set)**")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        
    with col_roc:
        st.write("**ROC Curve**")
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(metrics['FPR'], metrics['TPR'], label=f'AUC = {metrics["ROC-AUC"]:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)
        plt.close(fig_roc)

# --- TAB 4: Data Explorer ---
with tab4:
    st.subheader("Raw Dataset Explorer")
    st.write("Browse the first 50 rows of the Heart Disease dataset.")
    st.dataframe(df.head(50), use_container_width=True)
    
    st.write("**Dataset Statistics**")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.write("**Target Distribution**")
    fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x='target', palette='viridis', ax=ax_dist)
    ax_dist.set_xticklabels(['Low Risk (0)', 'High Risk (1)'])
    st.pyplot(fig_dist)
    plt.close(fig_dist)
