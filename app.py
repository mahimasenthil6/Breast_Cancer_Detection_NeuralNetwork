import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# 1. Page Configuration
st.set_page_config(page_title="Breast Cancer Detector", page_icon="🩺", layout="wide")
st.title("🩺 Full 30-Feature Breast Cancer Diagnostic System")
st.write("Input all 30 cellular measurements below to generate a real-time neural network diagnosis to detect cancer")

# 2. Load Your Saved Model and Scaler
@st.cache_resource
def load_assets():
    model = keras.models.load_model('breast_cancer_detector.keras')
    scaler = joblib.load('data_scaler.pkl')
    return model, scaler

try:
    loaded_model, loaded_scaler = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.header("🔬 Comprehensive Tumor Feature Profile")

# 3. Create 3 Columns for a Clean User Interface
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Mean Values")
    f1 = st.number_input("Mean Radius", min_value=0.0, value=17.99, key="f1")
    f2 = st.number_input("Mean Texture", min_value=0.0, value=10.38, key="f2")
    f3 = st.number_input("Mean Perimeter", min_value=0.0, value=122.8, key="f3")
    f4 = st.number_input("Mean Area", min_value=0.0, value=1001.0, key="f4")
    f5 = st.number_input("Mean Smoothness", min_value=0.0, value=0.1184, format="%.4f", key="f5")
    f6 = st.number_input("Mean Compactness", min_value=0.0, value=0.2776, format="%.4f", key="f6")
    f7 = st.number_input("Mean Concavity", min_value=0.0, value=0.3001, format="%.4f", key="f7")
    f8 = st.number_input("Mean Concave Points", min_value=0.0, value=0.1471, format="%.4f", key="f8")
    f9 = st.number_input("Mean Symmetry", min_value=0.0, value=0.2419, format="%.4f", key="f9")
    f10 = st.number_input("Mean Fractal Dimension", min_value=0.0, value=0.0787, format="%.4f", key="f10")

with col2:
    st.subheader("📐 Standard Error (SE)")
    f11 = st.number_input("Radius SE", min_value=0.0, value=1.095, key="f11")
    f12 = st.number_input("Texture SE", min_value=0.0, value=0.9053, key="f12")
    f13 = st.number_input("Perimeter SE", min_value=0.0, value=8.589, key="f13")
    f14 = st.number_input("Area SE", min_value=0.0, value=153.4, key="f14")
    f15 = st.number_input("Smoothness SE", min_value=0.0, value=0.0064, format="%.4f", key="f15")
    f16 = st.number_input("Compactness SE", min_value=0.0, value=0.0490, format="%.4f", key="f16")
    f17 = st.number_input("Concavity SE", min_value=0.0, value=0.0537, format="%.4f", key="f17")
    f18 = st.number_input("Concave Points SE", min_value=0.0, value=0.0159, format="%.4f", key="f18")
    f19 = st.number_input("Symmetry SE", min_value=0.0, value=0.0300, format="%.4f", key="f19")
    f20 = st.number_input("Fractal Dimension SE", min_value=0.0, value=0.0062, format="%.4f", key="f20")

with col3:
    st.subheader("🚨 Worst / Largest Values")
    f21 = st.number_input("Worst Radius", min_value=0.0, value=25.38, key="f21")
    f22 = st.number_input("Worst Texture", min_value=0.0, value=17.33, key="f22")
    f23 = st.number_input("Worst Perimeter", min_value=0.0, value=184.6, key="f23")
    f24 = st.number_input("Worst Area", min_value=0.0, value=2019.0, key="f24")
    f25 = st.number_input("Worst Smoothness", min_value=0.0, value=0.1622, format="%.4f", key="f25")
    f26 = st.number_input("Worst Compactness", min_value=0.0, value=0.6656, format="%.4f", key="f26")
    f27 = st.number_input("Worst Concavity", min_value=0.0, value=0.7119, format="%.4f", key="f27")
    f28 = st.number_input("Worst Concave Points", min_value=0.0, value=0.2654, format="%.4f", key="f28")
    f29 = st.number_input("Worst Symmetry", min_value=0.0, value=0.4601, format="%.4f", key="f29")
    f30 = st.number_input("Worst Fractal Dimension", min_value=0.0, value=0.1189, format="%.4f", key="f30")

# Collect all 30 actual user variables in the precise dataset order
raw_features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
                f21, f22, f23, f24, f25, f26, f27, f28, f29, f30]

# 4. Prediction Button Action
st.markdown("---")
if st.button("🚀 Run Comprehensive Neural Network Analysis", use_container_width=True):
    patient_array = np.array(raw_features).reshape(1, -1)
    scaled_data = loaded_scaler.transform(patient_array)
    prediction_prob = loaded_model.predict(scaled_data)
    final_class = np.argmax(prediction_prob)
    
    st.subheader("📊 Final Diagnostic Report:")
    if final_class == 0:
        st.error("### Result: MALIGNANT TUMOR DETECTED")
    else:
        st.success("### Result: BENIGN (NON-CANCEROUS) TUMOR")