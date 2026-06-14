import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import time

from neuroscan.inference import NeuroScanPipeline
from neuroscan.visualizer import plot_confidence_bar

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="NeuroScan Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM STYLING
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Geist:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    /* ——— Base Typography ——— */
    html, body, [class*="css"] {
        font-family: 'Geist', 'Plus Jakarta Sans', sans-serif;
        color: #e1e2eb;
    }

    /* ——— Hide Streamlit chrome ——— */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ——— Deep Space Background ——— */
    .stApp {
        background: #08090D;
    }

    /* ——— Sidebar styling ——— */
    section[data-testid="stSidebar"] {
        background: #0b0e14 !important;
        border-right: 1px solid rgba(180, 197, 255, 0.06) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #c3c6d6;
    }

    /* ——— Radio buttons as nav items ——— */
    div[data-testid="stRadio"] > div {
        gap: 0.25rem !important;
    }
    div[data-testid="stRadio"] > div > label {
        background: transparent !important;
        border: none !important;
        padding: 0.65rem 1rem !important;
        border-radius: 0.5rem !important;
        color: #8d909f !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stRadio"] > div > label:hover {
        background: rgba(50, 53, 60, 0.5) !important;
        color: #e1e2eb !important;
    }
    div[data-testid="stRadio"] > div > label[data-checked="true"] {
        background: rgba(180, 197, 255, 0.1) !important;
        color: #b4c5ff !important;
        border-right: 2px solid #b4c5ff !important;
        border-radius: 0.5rem 0 0 0.5rem !important;
    }

    /* ——— Hero Title ——— */
    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 48px;
        font-weight: 700;
        color: #b4c5ff;
        text-shadow: 0 0 20px rgba(180, 197, 255, 0.5);
        letter-spacing: -0.02em;
        text-align: center;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-family: 'Geist', sans-serif;
        font-size: 16px;
        color: rgba(195, 198, 214, 0.7);
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    /* ——— Glass Panel Cards ——— */
    .glass-panel {
        background: rgba(29, 32, 38, 0.4);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(180, 197, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.4s ease, border-color 0.4s ease;
    }
    .glass-panel:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 16px 48px rgba(180, 197, 255, 0.15), 0 0 30px rgba(180, 197, 255, 0.08);
        border-color: rgba(180, 197, 255, 0.25);
    }

    /* ——— Pathology Card ——— */
    .pathology-card {
        text-align: center;
        height: 340px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }
    .pathology-label {
        font-family: 'Geist', sans-serif;
        font-size: 12px;
        letter-spacing: 0.1em;
        font-weight: 600;
        text-transform: uppercase;
        color: #8d909f;
        margin-bottom: 1.5rem;
    }
    .pathology-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 56px;
        font-weight: 700;
        color: #b4c5ff;
        text-shadow: 0 0 20px rgba(180, 197, 255, 0.5);
        letter-spacing: -0.02em;
        line-height: 1;
        margin-bottom: 1rem;
    }
    .badge-critical {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 9999px;
        background: rgba(255, 180, 171, 0.1);
        border: 1px solid rgba(255, 180, 171, 0.2);
        color: #ffb4ab;
        font-family: 'Geist', sans-serif;
        font-size: 10px;
        letter-spacing: 0.1em;
        font-weight: 600;
        animation: pulse-badge 2s ease-in-out infinite;
    }
    .badge-stable {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 9999px;
        background: rgba(40, 217, 243, 0.1);
        border: 1px solid rgba(40, 217, 243, 0.2);
        color: #28d9f3;
        font-family: 'Geist', sans-serif;
        font-size: 10px;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    @keyframes pulse-badge {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* ——— Confidence Card ——— */
    .confidence-card {
        text-align: center;
        height: 340px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .confidence-label {
        font-family: 'Geist', sans-serif;
        font-size: 12px;
        letter-spacing: 0.1em;
        font-weight: 600;
        text-transform: uppercase;
        color: #8d909f;
        margin-bottom: 1rem;
    }
    .confidence-value {
        font-family: 'Geist', sans-serif;
        font-size: 48px;
        font-weight: 700;
        color: #b4c5ff;
        text-shadow: 0 0 20px rgba(180, 197, 255, 0.5);
        letter-spacing: -0.01em;
    }
    .confidence-ai-label {
        font-family: 'Geist', sans-serif;
        font-size: 10px;
        letter-spacing: 0.1em;
        font-weight: 600;
        color: #28d9f3;
        margin-top: 4px;
    }
    .meta-footer {
        margin-top: 1rem;
        display: flex;
        gap: 1.5rem;
        font-family: 'Geist', sans-serif;
        font-size: 10px;
        letter-spacing: 0.1em;
        font-weight: 600;
        color: rgba(141, 144, 159, 0.4);
    }

    /* ——— Analysis Summary Card ——— */
    .summary-card {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    .summary-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #e1e2eb;
    }
    .summary-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(67, 70, 84, 0.3);
    }
    .summary-row:last-of-type {
        border-bottom: none;
    }
    .summary-field-label {
        font-family: 'Geist', sans-serif;
        font-size: 10px;
        letter-spacing: 0.1em;
        font-weight: 600;
        color: #8d909f;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .summary-field-value {
        font-size: 14px;
        font-weight: 600;
        color: #e1e2eb;
    }

    /* ——— File Uploader ——— */
    div[data-testid="stFileUploader"] > section {
        padding: 2rem;
        border-radius: 16px;
        background: rgba(29, 32, 38, 0.4);
        border: 2px dashed rgba(180, 197, 255, 0.2) !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: rgba(180, 197, 255, 0.5) !important;
        background: rgba(29, 32, 38, 0.6);
    }

    /* ——— Tabs styling ——— */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(29, 32, 38, 0.4);
        border-radius: 12px;
        padding: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #8d909f !important;
        padding: 0.5rem 1rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(180, 197, 255, 0.1) !important;
        color: #b4c5ff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ——— Metrics styling ——— */
    [data-testid="stMetric"] {
        background: rgba(29, 32, 38, 0.4);
        border: 1px solid rgba(180, 197, 255, 0.1);
        border-radius: 16px;
        padding: 1rem 1.5rem;
    }

    /* ——— Dividers ——— */
    hr {
        border-color: rgba(67, 70, 84, 0.2) !important;
        margin: 2rem 0 !important;
    }

    /* ——— Section Labels ——— */
    .section-label {
        font-family: 'Geist', sans-serif;
        font-size: 12px;
        letter-spacing: 0.1em;
        font-weight: 600;
        text-transform: uppercase;
        color: #8d909f;
        margin-bottom: 1rem;
    }

    /* ——— Enhanced Image Containers ——— */
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.1);
        background: rgba(15, 23, 42, 0.6);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.3s ease;
        margin-bottom: 0.5rem;
    }
    
    .image-container:hover {
        transform: scale(1.03);
        border-color: rgba(180, 197, 255, 0.4);
        box-shadow: 0 8px 25px rgba(180, 197, 255, 0.25);
    }

    .image-caption {
        text-align: center;
        padding: 0.8rem;
        background: rgba(15, 23, 42, 0.8);
        color: #e2e8f0;
        font-family: 'Geist', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* MRI Viewport Header */
    .viewport-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1rem;
    }
    .viewport-header .material-symbols-outlined {
        color: #b4c5ff;
        font-size: 24px;
    }
    .viewport-header span:last-child {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: #e1e2eb;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# INITIALIZE PIPELINE
# =====================================================
@st.cache_resource
def load_pipeline():
    return NeuroScanPipeline()

pipeline = load_pipeline()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 24px; font-weight: 700; color: #b4c5ff; text-shadow: 0 0 15px rgba(180,197,255,0.4); margin-bottom: 0;">NeuroScan</h1>
        <p style="font-family: 'Geist', sans-serif; font-size: 14px; color: rgba(195,198,214,0.7); margin-top: 4px;">Precision Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["🔬 Live Scan", "🕒 History", "📂 Patient Records", "⚙️ Settings"], label_visibility="collapsed")

    st.markdown("---")

    st.markdown("**System Status**")
    if pipeline.seg_model is not None:
        st.success("✅ Segmentation Core Online")
    else:
        st.error("❌ Segmentation Core Offline")

    if pipeline.cls_model is not None:
        st.success("✅ Classification Node Online")
    else:
        st.error("❌ Classification Node Offline")

# =====================================================
# HELPER: Build SVG confidence ring
# =====================================================
def build_confidence_card(confidence_pct, latency_ms):
    """Creates the full confidence card HTML with SVG donut chart."""
    radius = 45
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - confidence_pct / 100)
    return (
        '<div class="glass-panel confidence-card">'
        '<span class="confidence-label">Diagnostic Confidence</span>'
        '<div style="position:relative;width:200px;height:200px;margin:0 auto;">'
        '<svg viewBox="0 0 100 100" style="width:100%;height:100%;transform:rotate(-90deg);">'
        f'<circle cx="50" cy="50" r="{radius}" fill="transparent" stroke="rgba(255,255,255,0.05)" stroke-width="8"/>'
        f'<circle cx="50" cy="50" r="{radius}" fill="transparent" stroke="url(#ng)" stroke-width="8" '
        f'stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}" stroke-linecap="round"/>'
        '<defs><linearGradient id="ng" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" style="stop-color:#54339c"/>'
        '<stop offset="50%" style="stop-color:#2355cc"/>'
        '<stop offset="100%" style="stop-color:#28d9f3"/>'
        '</linearGradient></defs>'
        '</svg>'
        '<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">'
        f'<span class="confidence-value">{confidence_pct:.1f}%</span>'
        '<span class="confidence-ai-label">CONFIRMED AI OUTPUT</span>'
        '</div>'
        '</div>'
        '<div style="margin-top:1rem;display:flex;gap:1.5rem;justify-content:center;'
        'font-family:Geist,sans-serif;font-size:10px;letter-spacing:0.1em;font-weight:600;color:rgba(141,144,159,0.4);">'
        f'<span>MODEL: EFFNET-V2B0</span><span>LATENCY: {latency_ms}MS</span>'
        '</div>'
        '</div>'
    )

# =====================================================
# HELPER: Build severity badge
# =====================================================
def get_severity_badge(pred_class):
    """Returns HTML badge based on pathology type."""
    critical_classes = ['glioma', 'meningioma']
    if pred_class.lower() in critical_classes:
        return '<div class="badge-critical"><span class="material-symbols-outlined" style="font-size:14px;">warning</span>CRITICAL FINDING</div>'
    else:
        return '<div class="badge-stable"><span class="material-symbols-outlined" style="font-size:14px;">check_circle</span>STABLE</div>'

# =====================================================
# HELPER: Analysis summary descriptions
# =====================================================
def get_analysis_summary(pred_class):
    """Returns contextual analysis summary based on prediction."""
    summaries = {
        'glioma': {
            'morphology': 'Intra-axial mass',
            'localization': 'Cerebral hemisphere',
            'vascularity': 'High hyperintensity'
        },
        'meningioma': {
            'morphology': 'Extra-axial mass',
            'localization': 'Frontal Lobe Sulcus',
            'vascularity': 'Moderate hyperintensity'
        },
        'pituitary': {
            'morphology': 'Sellar mass',
            'localization': 'Pituitary fossa',
            'vascularity': 'Mild enhancement'
        },
        'notumor': {
            'morphology': 'No anomaly detected',
            'localization': 'N/A',
            'vascularity': 'Normal signal intensity'
        }
    }
    key = pred_class.lower().replace(' ', '')
    return summaries.get(key, summaries['notumor'])

# =====================================================
# PAGES
# =====================================================

if page == "🔬 Live Scan":
    st.markdown('<div class="hero-title">NeuroScan Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Automated MRI Pathology Identification via Attention U-Net & EfficientNetV2B0</div>', unsafe_allow_html=True)

    if pipeline.seg_model is None or pipeline.cls_model is None:
        st.warning("⚠️ Models not loaded. Run the training scripts to generate `.keras` checkpoints in `checkpoints/`.")

    st.markdown('<p class="section-label">Insert Medical Imaging Data (JPG/PNG)</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None and pipeline.seg_model is not None and pipeline.cls_model is not None:
        pil_img = Image.open(uploaded_file)
        open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Animated progress bar
        progress_bar = st.progress(0, text="Initializing Neural Pipeline...")
        time.sleep(0.4)
        progress_bar.progress(30, text="Segmenting Anomaly Regions (Attention U-Net)...")
        time.sleep(0.6)
        progress_bar.progress(60, text="Extracting Region of Interest...")

        start_time = time.time()
        results = pipeline.process_image(open_cv_image)
        latency = int((time.time() - start_time) * 1000)

        progress_bar.progress(90, text="Classifying Pathology (EfficientNetV2B0)...")
        time.sleep(0.4)
        progress_bar.progress(100, text="Generating Grad-CAM Explanations...")
        time.sleep(0.3)
        progress_bar.empty()

        st.toast('Neural processing complete!', icon='🧠')

        confidence_pct = results['confidence'] * 100
        pred_class = results['pred_class']

        st.markdown("---")

        # ——— Results Bento Grid: Pathology + Confidence ———
        col_path, col_conf = st.columns(2, gap="large")

        with col_path:
            badge_html = get_severity_badge(pred_class)
            st.markdown(f"""
            <div class="glass-panel pathology-card">
                <span class="pathology-label">Detected Pathology</span>
                <div class="pathology-value">{pred_class.upper()}</div>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

        with col_conf:
            conf_card = build_confidence_card(confidence_pct, latency)
            st.markdown(conf_card, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ——— Middle Section: Spatial Diagnostics (2x2 Grid) ———
        st.markdown("""
        <div class="viewport-header">
            <span class="material-symbols-outlined">visibility</span>
            <span>Axial MRI Layer Scan</span>
        </div>
        """, unsafe_allow_html=True)

        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row2_col1, row2_col2 = st.columns(2, gap="medium")

        raw_rgb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        target_dim = (raw_rgb.shape[1], raw_rgb.shape[0])

        with row1_col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(raw_rgb, use_container_width=True)
            st.markdown('<div class="image-caption">Raw MRI Scan</div></div>', unsafe_allow_html=True)
        
        with row1_col2:
            overlay_resized = cv2.resize(results['overlay'], target_dim, interpolation=cv2.INTER_CUBIC)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(overlay_resized, use_container_width=True)
            st.markdown('<div class="image-caption">Segmentation Mask Overlay</div></div>', unsafe_allow_html=True)
        
        with row2_col1:
            roi_display = results['cropped_roi']
            if len(roi_display.shape) == 2:
                roi_display = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2RGB)
            else:
                roi_display = cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_display, target_dim, interpolation=cv2.INTER_CUBIC)
            
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(roi_resized, use_container_width=True)
            st.markdown('<div class="image-caption">Isolated Region of Interest</div></div>', unsafe_allow_html=True)
            
        with row2_col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            if results.get('gradcam_overlay') is not None:
                gradcam = results['gradcam_overlay']
                gradcam_resized = cv2.resize(gradcam, target_dim, interpolation=cv2.INTER_CUBIC)
                st.image(gradcam_resized, use_container_width=True)
            else:
                st.image(roi_resized, use_container_width=True)
            st.markdown('<div class="image-caption">Grad-CAM (Model Focus)</div></div>', unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ——— Bottom Section: Probability Distribution + Analysis Summary ———
        col_prob, col_summary = st.columns([3, 1], gap="large")

        with col_prob:
            st.markdown('<p class="section-label">Probability Distribution</p>', unsafe_allow_html=True)
            fig = plot_confidence_bar(results['probabilities'])
            st.plotly_chart(fig, use_container_width=True)

        with col_summary:
            st.markdown('<p class="section-label">Clinical Details</p>', unsafe_allow_html=True)
            summary = get_analysis_summary(pred_class)
            st.markdown(f"""
            <div class="glass-panel summary-card" style="height: 380px;">
                <div class="summary-title">Analysis Summary</div>
                <div class="summary-row">
                    <div>
                        <div class="summary-field-label">MORPHOLOGY</div>
                        <div class="summary-field-value">{summary['morphology']}</div>
                    </div>
                    <span class="material-symbols-outlined" style="color: #28d9f3;">check_circle</span>
                </div>
                <div class="summary-row">
                    <div>
                        <div class="summary-field-label">LOCALIZATION</div>
                        <div class="summary-field-value">{summary['localization']}</div>
                    </div>
                    <span class="material-symbols-outlined" style="color: #28d9f3;">check_circle</span>
                </div>
                <div class="summary-row" style="border-bottom:none;">
                    <div>
                        <div class="summary-field-label">VASCULARITY</div>
                        <div class="summary-field-value">{summary['vascularity']}</div>
                    </div>
                    <span class="material-symbols-outlined" style="color: rgba(141,144,159,0.4);">info</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "📄 Generate Clinical Report",
                data=f"NeuroScan Clinical Report\n{'='*40}\nPathology: {pred_class.upper()}\nConfidence: {confidence_pct:.1f}%\nMorphology: {summary['morphology']}\nLocalization: {summary['localization']}\nVascularity: {summary['vascularity']}\nModel: EfficientNetV2B0\nLatency: {latency}ms\n",
                file_name=f"neuroscan_report_{pred_class}.txt",
                mime="text/plain",
                use_container_width=True,
                type="primary"
            )

        if np.sum(results['mask']) == 0:
            st.info("ℹ️ Minimal anomalous tissue detected. Region of Interest defaults to the full frame.")

elif page == "🕒 History":
    st.markdown('<div class="hero-title">Analysis History</div>', unsafe_allow_html=True)
    st.info("No past scans found for this session. History tracking module is under development.")

elif page == "📂 Patient Records":
    st.markdown('<div class="hero-title">Patient Records</div>', unsafe_allow_html=True)
    st.info("Patient database connection is currently offline. Please configure EHR integration.")

elif page == "⚙️ Settings":
    st.markdown('<div class="hero-title">System Settings</div>', unsafe_allow_html=True)
    st.info("Confidence Threshold and Model Selection settings will be available in the next update.")
