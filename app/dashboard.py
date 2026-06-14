import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

from neuroscan.inference import NeuroScanPipeline
from neuroscan.visualizer import plot_confidence_bar

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM STYLING
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide Streamlit Default elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Premium Dark Theme Background */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e293b 0%, #0b0f19 50%, #000000 100%);
        color: #e2e8f0;
    }

    /* Enhanced Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0ea5e9, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        padding-top: 2rem;
        text-shadow: 0px 4px 15px rgba(99, 102, 241, 0.3);
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px rgba(99,102,241,0.2), 0 0 20px rgba(99,102,241,0.2); }
        to { text-shadow: 0 0 20px rgba(99,102,241,0.5), 0 0 30px rgba(168,85,247,0.4); }
    }

    .sub-header {
        font-size: 1.1rem;
        font-weight: 300;
        text-align: center;
        color: #94a3b8;
        margin-bottom: 3rem;
        letter-spacing: 1px;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #cbd5e1;
        font-weight: 600;
    }

    /* Enhanced Image Containers */
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.1);
        background: rgba(15, 23, 42, 0.6);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.03);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25);
    }

    .image-caption {
        text-align: center;
        padding: 0.8rem;
        background: linear-gradient(to top, rgba(15, 23, 42, 1), rgba(15, 23, 42, 0.8));
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Custom Styling for Streamlit elements */
    div[data-testid="stFileUploader"] > section {
        padding: 2rem;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.3);
        border: 1px dashed rgba(148, 163, 184, 0.4);
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: #818cf8;
        background: rgba(49, 46, 129, 0.2);
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 3rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# INITIALIZE PIPELINE
# =====================================================
@st.cache_resource
def load_pipeline():
    # Use environment vars or default config paths.
    # We allow it to fail gracefully if files are missing.
    return NeuroScanPipeline()

pipeline = load_pipeline()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### 🎛️ NeuroScan Control Panel")
    st.info("Upload a patient's MRI scan to begin the autonomous analysis workflow.")
    
    st.markdown("---")
    st.markdown("**Status Checker:**")
    if pipeline.seg_model is not None:
        st.success("✅ Segmentation Core Active")
    else:
        st.error("❌ Segmentation Core Offline (Weights not found)")
        
    if pipeline.cls_model is not None:
        st.success("✅ Classification Node Active")
    else:
        st.error("❌ Classification Node Offline (Weights not found)")

# =====================================================
# MAIN DASHBOARD
# =====================================================
st.markdown('<div class="main-header">NeuroScan Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated MRI Pathology Identification via Attention U-Net & EfficientNetV2B0</div>', unsafe_allow_html=True)

if pipeline.seg_model is None or pipeline.cls_model is None:
    st.warning("⚠️ Models have not been trained or configured correctly. Please run the training scripts to generate `.keras` checkpoints in the `checkpoints/` directory.")

uploaded_file = st.file_uploader("Insert Medical Imaging Data (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and pipeline.seg_model is not None and pipeline.cls_model is not None:
    # Read Image
    pil_img = Image.open(uploaded_file)
    # Convert to BGR format which cv2 uses internally and our pipeline expects
    open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing scan morphology..."):
        results = pipeline.process_image(open_cv_image)

    st.markdown("---")
    
    # Hero Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Detected Pathology</div>
            <div class="metric-value">{results['pred_class'].upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Diagnostic Confidence</div>
            <div class="metric-value">{results['confidence']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Visual Diagnostics
    st.markdown("### 🔬 Spatial Diagnostics & Explainability")
    c1, c2, c3, c4 = st.columns(4)

    # Convert cv2 images (BGR/gray) to RGB for Streamlit rendering
    raw_rgb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    
    with c1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(raw_rgb, use_container_width=True)
        st.markdown('<div class="image-caption">Raw MRI Scan</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(results['overlay'], use_container_width=True)
        st.markdown('<div class="image-caption">Segmentation Mask Overlay</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        roi_display = results['cropped_roi']
        if len(roi_display.shape) == 2:
            roi_display = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2RGB)
        else:
            roi_display = cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB)
            
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(roi_display, use_container_width=True)
        st.markdown('<div class="image-caption">Isolated Region of Interest</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c4:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        if results.get('gradcam_overlay') is not None:
            st.image(results['gradcam_overlay'], use_container_width=True)
        else:
            st.image(roi_display, use_container_width=True) # Fallback
        st.markdown('<div class="image-caption">Grad-CAM (Model Focus)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Confidence Chart
    st.markdown("### 📊 Probability Distribution")
    fig = plot_confidence_bar(results['probabilities'])
    st.plotly_chart(fig, use_container_width=True)
    
    if np.sum(results['mask']) == 0:
        st.info("ℹ️ Minimal to no anomalous tissue detected by the segmentation module. The Region of Interest defaults to the full frame.")
