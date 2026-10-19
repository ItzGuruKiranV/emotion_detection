"""
AI-Powered Multimodal Emotion Analysis Platform.
Enterprise-grade Streamlit application integrating Vision, Acoustic SER, NLP,
Multimodal Late Fusion, History Analytics, and Real-time Evaluation.
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
import io
import time
import matplotlib.pyplot as plt

# Configuration and Backend Modules
from src.config import (
    CANONICAL_EMOTIONS,
    EMOTION_COLORS,
    EMOTION_EMOJIS,
    DEFAULT_MODALITY_WEIGHTS,
    DATA_SAMPLE_DIR
)
from src.vision.classifier import VisionEmotionClassifier
from src.audio.classifier import AudioEmotionClassifier
from src.text.classifier import TextEmotionClassifier
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.database.models import (
    add_prediction,
    get_history,
    get_emotion_distribution,
    clear_history
)
from src.evaluation.evaluator import compute_metrics, compute_latency_stats
from src.utils.validation import ValidationError

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Emotion AI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Dark Glassmorphism Aesthetics
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(22, 27, 34, 0.75);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    
    .conflict-alert {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px;
        color: #fca5a5;
        margin-top: 10px;
    }
    
    .consensus-badge {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 12px;
        color: #6ee7b7;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Vision Engine (MobileNetV2)...")
def get_vision_engine():
    return VisionEmotionClassifier.get_instance()


@st.cache_resource(show_spinner="Initializing Acoustic Audio Engine (SER)...")
def get_audio_engine():
    return AudioEmotionClassifier.get_instance()


@st.cache_resource(show_spinner="Initializing NLP Engine (BERT)...")
def get_text_engine():
    return TextEmotionClassifier.get_instance()


@st.cache_resource
def get_fusion_engine():
    return MultimodalFusionEngine()


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/clouds/200/brain.png", width=90)
st.sidebar.title("Multimodal Emotion AI")
st.sidebar.caption("v2.0.0 | Vision • Audio • NLP • Late Fusion")

navigation_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "🌟 Platform Overview",
        "📷 Facial Emotion Analysis",
        "🎙️ Acoustic Audio Emotion",
        "📝 Text Emotion Analysis",
        "🧬 Multimodal Fusion Studio",
        "📊 Prediction Analytics",
        "🔬 Model Evaluation & Benchmarks",
        "⚙️ Architecture & Model Registry"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Ethical AI Disclosure:** This platform estimates observable facial expressions, "
    "acoustic vocal dynamics, and textual sentiment categories. It is not a psychological or medical diagnostic tool."
)


# ---------------------------------------------------------------------------
# Utility Plotting Functions
# ---------------------------------------------------------------------------
def plot_probabilities(probabilities: dict, title: str = "Emotion Probabilities"):
    """Render horizontal bar chart for emotion probabilities."""
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    emotions = [k.capitalize() for k, v in sorted_items]
    scores = [v * 100 for k, v in sorted_items]
    colors = [EMOTION_COLORS.get(k.lower(), "#3b82f6") for k, v in sorted_items]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    bars = ax.barh(emotions[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", color="#8b949e", fontsize=9)
    ax.set_title(title, color="#e6edf3", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("#30363d")
    ax.spines['bottom'].set_color("#30363d")

    for bar in bars:
        w = bar.get_width()
        if w > 2:
            ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                    ha='left', va='center', color="#e6edf3", fontsize=8, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# 1. PLATFORM OVERVIEW
# ===========================================================================
if navigation_mode == "🌟 Platform Overview":
    st.title("🧠 AI-Powered Multimodal Emotion Analysis Platform")
    st.markdown("### Next-Generation Tri-Modal Emotional Intelligence Architecture")

    st.markdown("""
    Welcome to the enterprise-grade **Multimodal Emotion Detection & Analytics Platform**. 
    This system combines computer vision, speech emotion acoustics, and transformer NLP 
    to provide robust emotional perception under noisy, real-world conditions.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📷 Vision Modality</h3>
            <p><strong>MobileNetV2</strong> fine-tuned on facial expressions. Real-time face detection with Haar Cascades and multi-face classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎙️ Audio Modality</h3>
            <p><strong>Acoustic SER Engine</strong> analyzing prosodic pitch (F0), RMS dynamics, zero-crossing rate, and spectral MFCC envelopes.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📝 Text Modality</h3>
            <p><strong>BERT Sequence Classifier</strong> fine-tuned for emotional sentiment, mapping fine-grained intent to canonical Ekman categories.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🧬 Late Fusion</h3>
            <p><strong>Quality-Aware Late Fusion</strong> with dynamic reliability weighting, Shannon entropy uncertainty, and cross-modal conflict alerts.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💡 Why Multimodal Emotion AI?")
    st.write("""
    Single-modality emotion detection frequently breaks down:
    - **Visual smiles** can mask negative internal emotions or polite social conventions.
    - **Acoustic sarcasm** (saying 'Great job' in an aggressive tone) inverts semantic text meaning.
    - **Text alone** lacks vocal prosody, pitch inflections, and non-verbal cues.
    
    By fusing **Vision + Speech Acoustics + Text**, our late fusion engine cross-validates emotional cues,
    detects cross-modal contradictions, and reports uncertainty when signals disagree.
    """)


# ===========================================================================
# 2. FACIAL EMOTION ANALYSIS
# ===========================================================================
elif navigation_mode == "📷 Facial Emotion Analysis":
    st.title("📷 Facial Emotion Classification")
    st.caption("Computer Vision pipeline: Face Detection • Region Localization • MobileNetV2 Expression Classification")

    vision_engine = get_vision_engine()

    col_input, col_result = st.columns([1, 1])

    with col_input:
        input_method = st.radio("Image Input Mode", ["Upload Image", "Webcam Snapshot", "Load Sample Asset"], horizontal=True)

        img_to_process = None

        if input_method == "Upload Image":
            uploaded = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])
            if uploaded:
                img_to_process = Image.open(uploaded).convert("RGB")
                st.image(img_to_process, caption="Uploaded Image", use_container_width=True)

        elif input_method == "Webcam Snapshot":
            camera_img = st.camera_input("Take a portrait photo")
            if camera_img:
                img_to_process = Image.open(camera_img).convert("RGB")

        else:
            sample_dir = DATA_SAMPLE_DIR / "images"
            sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
            if sample_files:
                choice = st.selectbox("Select Sample Asset", [f.name for f in sample_files])
                img_to_process = Image.open(sample_dir / choice).convert("RGB")
                st.image(img_to_process, caption=f"Sample: {choice}", use_container_width=True)
            else:
                st.warning("Sample assets not found. Run `scripts/create_sample_assets.py`.")

        analyze_btn = st.button("🔍 Analyze Facial Expression", type="primary", disabled=(img_to_process is None))

    with col_result:
        if analyze_btn and img_to_process:
            with st.spinner("Executing Face Localization and MobileNetV2 Inference..."):
                res = vision_engine.analyze_image(img_to_process, annotate=True)

                # Persist to database
                add_prediction(
                    modality="vision",
                    dominant_emotion=res["dominant_emotion"],
                    confidence=res["confidence"],
                    probabilities=res["probabilities"],
                    metadata={"faces_detected": res["faces_detected"], "latency_ms": res["latency_ms"]}
                )

            st.subheader("Analysis Results")
            emoji = EMOTION_EMOJIS.get(res["dominant_emotion"], "😐")
            st.markdown(f"### {emoji} Dominant Emotion: **{res['dominant_emotion'].upper()}**")
            st.metric("Model Confidence", f"{res['confidence'] * 100:.1f}%", f"Latency: {res['latency_ms']} ms")

            st.write(f"Faces Detected: **{res['faces_detected']}**")

            if res["annotated_image"]:
                st.image(res["annotated_image"], caption="Annotated Faces & Bounding Boxes", use_container_width=True)

            plot_probabilities(res["probabilities"], "Facial Expression Probabilities")


# ===========================================================================
# 3. ACOUSTIC AUDIO EMOTION
# ===========================================================================
elif navigation_mode == "🎙️ Acoustic Audio Emotion":
    st.title("🎙️ Acoustic Speech Emotion Recognition (SER)")
    st.caption("Acoustic Feature Extraction: Mel-Spectrogram • MFCC • Pitch (F0) • Energy Dynamics")

    audio_engine = get_audio_engine()

    col_input, col_result = st.columns([1, 1])

    with col_input:
        input_choice = st.radio("Audio Source", ["Upload Audio File", "Load Sample Asset"], horizontal=True)

        audio_input = None
        audio_name = "User Audio"

        if input_choice == "Upload Audio File":
            uploaded_audio = st.file_uploader("Upload audio (WAV, MP3)", type=["wav", "mp3", "ogg"])
            if uploaded_audio:
                audio_input = uploaded_audio.read()
                audio_name = uploaded_audio.name
                st.audio(audio_input)
        else:
            sample_dir = DATA_SAMPLE_DIR / "audio"
            sample_files = list(sample_dir.glob("*.wav")) if sample_dir.exists() else []
            if sample_files:
                selected_sample = st.selectbox("Select Sample Audio Clip", [f.name for f in sample_files])
                audio_path = sample_dir / selected_sample
                with open(audio_path, "rb") as f:
                    audio_input = f.read()
                audio_name = selected_sample
                st.audio(audio_input)
            else:
                st.warning("Sample audio not found. Run `scripts/create_sample_assets.py`.")

        analyze_audio_btn = st.button("🎧 Analyze Audio Emotion", type="primary", disabled=(audio_input is None))

    with col_result:
        if analyze_audio_btn and audio_input:
            with st.spinner("Extracting Acoustic Features & Classifying..."):
                try:
                    res = audio_engine.analyze_audio(audio_input)

                    add_prediction(
                        modality="audio",
                        dominant_emotion=res["dominant_emotion"],
                        confidence=res["confidence"],
                        probabilities=res["probabilities"],
                        metadata={"duration_sec": res["duration_sec"], "latency_ms": res["latency_ms"]}
                    )

                    emoji = EMOTION_EMOJIS.get(res["dominant_emotion"], "🎙️")
                    st.markdown(f"### {emoji} Acoustic Emotion: **{res['dominant_emotion'].upper()}**")
                    st.metric("Acoustic Confidence", f"{res['confidence'] * 100:.1f}%", f"Latency: {res['latency_ms']} ms")

                    # Acoustic Indicators Display
                    stats = res["acoustic_summary"]
                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.metric("Pitch (F0)", f"{stats['mean_pitch_hz']} Hz")
                    col_s2.metric("RMS Energy", f"{stats['mean_rms_energy']:.3f}")
                    col_s3.metric("Duration", f"{res['duration_sec']:.2f} s")

                    # Mel Spectrogram Plot
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    fig.patch.set_facecolor("#161b22")
                    ax.set_facecolor("#161b22")
                    im = ax.imshow(res["mel_spectrogram_db"], aspect='auto', origin='lower', cmap='magma')
                    ax.set_title("Log-Mel Spectrogram", color="#e6edf3", fontsize=10)
                    ax.set_xlabel("Time Frames", color="#8b949e", fontsize=8)
                    ax.set_ylabel("Mel Frequency Bins", color="#8b949e", fontsize=8)
                    ax.tick_params(colors="#8b949e", labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    plot_probabilities(res["probabilities"], "Acoustic Emotion Probabilities")

                except ValidationError as ve:
                    st.error(f"Validation Error: {ve}")


# ===========================================================================
# 4. TEXT EMOTION ANALYSIS
# ===========================================================================
elif navigation_mode == "📝 Text Emotion Analysis":
    st.title("📝 Text Emotion Classification")
    st.caption("Transformer NLP Engine: BERT sequence classification fine-tuned on emotion taxonomy")

    text_engine = get_text_engine()

    col_input, col_result = st.columns([1, 1])

    with col_input:
        sample_choice = st.selectbox(
            "Quick Benchmark Presets",
            [
                "Custom Text...",
                "I am absolutely thrilled and overjoyed with this incredible milestone!",
                "I feel so devastated, alone, and completely heartbroken today.",
                "I am furious that our work was destroyed without any explanation!",
                "I am terrified and panicking about what will happen next.",
                "The project architecture discussion is scheduled tomorrow morning at 10 AM."
            ]
        )

        default_text = "" if sample_choice == "Custom Text..." else sample_choice
        user_text = st.text_area("Input Text for Emotion Analysis", value=default_text, height=140)

        analyze_text_btn = st.button("📖 Analyze Text Emotion", type="primary", disabled=(not user_text.strip()))

    with col_result:
        if analyze_text_btn and user_text.strip():
            with st.spinner("Tokenizing & Running BERT Sequence Classification..."):
                try:
                    res = text_engine.analyze_text(user_text)

                    add_prediction(
                        modality="text",
                        dominant_emotion=res["dominant_emotion"],
                        confidence=res["confidence"],
                        probabilities=res["probabilities"],
                        metadata={"char_count": res["char_count"], "word_count": res["word_count"], "latency_ms": res["latency_ms"]}
                    )

                    emoji = EMOTION_EMOJIS.get(res["dominant_emotion"], "📝")
                    st.markdown(f"### {emoji} Text Emotion: **{res['dominant_emotion'].upper()}**")
                    st.metric("Text Confidence", f"{res['confidence'] * 100:.1f}%", f"Latency: {res['latency_ms']} ms")

                    st.caption(f"Tokens/Words: {res['word_count']} words ({res['char_count']} characters)")
                    plot_probabilities(res["probabilities"], "Text Emotion Distribution")

                except ValidationError as ve:
                    st.error(f"Validation Error: {ve}")


# ===========================================================================
# 5. MULTIMODAL FUSION STUDIO
# ===========================================================================
elif navigation_mode == "🧬 Multimodal Fusion Studio":
    st.title("🧬 Multimodal Late Fusion Studio")
    st.caption("Simultaneous Multi-Sensor Perception: Combines Vision, Speech Audio, and Text with Conflict Detection")

    fusion_engine = get_fusion_engine()
    vision_engine = get_vision_engine()
    audio_engine = get_audio_engine()
    text_engine = get_text_engine()

    st.subheader("1. Modality Inputs")
    col_v, col_a, col_t = st.columns(3)

    with col_v:
        st.markdown("#### 📷 Vision Input")
        v_opt = st.radio("Vision Source", ["None", "Sample Happy Face", "Sample Sad Face", "Upload Face"], key="fuse_v")
        v_image = None
        if v_opt == "Sample Happy Face":
            p = DATA_SAMPLE_DIR / "images" / "sample_happy_face.jpg"
            if p.exists():
                v_image = Image.open(p)
                st.image(v_image, width=150)
        elif v_opt == "Sample Sad Face":
            p = DATA_SAMPLE_DIR / "images" / "sample_sad_face.jpg"
            if p.exists():
                v_image = Image.open(p)
                st.image(v_image, width=150)
        elif v_opt == "Upload Face":
            up_f = st.file_uploader("Upload Image", type=["jpg", "png"], key="fuse_v_up")
            if up_f:
                v_image = Image.open(up_f).convert("RGB")
                st.image(v_image, width=150)

    with col_a:
        st.markdown("#### 🎙️ Audio Input")
        a_opt = st.radio("Audio Source", ["None", "Sample Happy Tone", "Sample Angry Tone", "Sample Sad Tone", "Upload Audio"], key="fuse_a")
        a_bytes = None
        if a_opt == "Sample Happy Tone":
            p = DATA_SAMPLE_DIR / "audio" / "sample_happy_tone.wav"
            if p.exists():
                with open(p, "rb") as f:
                    a_bytes = f.read()
                st.audio(a_bytes)
        elif a_opt == "Sample Angry Tone":
            p = DATA_SAMPLE_DIR / "audio" / "sample_angry_tone.wav"
            if p.exists():
                with open(p, "rb") as f:
                    a_bytes = f.read()
                st.audio(a_bytes)
        elif a_opt == "Sample Sad Tone":
            p = DATA_SAMPLE_DIR / "audio" / "sample_sad_tone.wav"
            if p.exists():
                with open(p, "rb") as f:
                    a_bytes = f.read()
                st.audio(a_bytes)
        elif a_opt == "Upload Audio":
            up_a = st.file_uploader("Upload Audio", type=["wav", "mp3"], key="fuse_a_up")
            if up_a:
                a_bytes = up_a.read()
                st.audio(a_bytes)

    with col_t:
        st.markdown("#### 📝 Text Input")
        t_preset = st.selectbox(
            "Text Scenario",
            [
                "None",
                "I am absolutely thrilled and overjoyed with this incredible milestone!",
                "I feel so devastated, alone, and completely heartbroken today.",
                "I am furious that our work was destroyed without any explanation!",
                "Custom Text..."
            ],
            key="fuse_t_select"
        )
        t_text = None
        if t_preset == "Custom Text...":
            custom_t = st.text_area("Enter sentence", "", key="fuse_t_custom")
            if custom_t.strip():
                t_text = custom_t.strip()
        elif t_preset != "None":
            t_text = t_preset
            st.info(f'"{t_text}"')

    st.markdown("---")
    st.subheader("2. Fusion Prior Weights (Optional Tuning)")
    w_col1, w_col2, w_col3 = st.columns(3)
    w_v = w_col1.slider("Vision Weight Prior", 0.0, 1.0, DEFAULT_MODALITY_WEIGHTS["vision"], 0.05)
    w_a = w_col2.slider("Audio Weight Prior", 0.0, 1.0, DEFAULT_MODALITY_WEIGHTS["audio"], 0.05)
    w_t = w_col3.slider("Text Weight Prior", 0.0, 1.0, DEFAULT_MODALITY_WEIGHTS["text"], 0.05)

    fuse_trigger = st.button("🚀 Execute Multimodal Fusion Analysis", type="primary")

    if fuse_trigger:
        active_inputs = {}
        if v_image:
            with st.spinner("Analyzing Vision modality..."):
                active_inputs["vision"] = vision_engine.analyze_image(v_image, annotate=False)
        if a_bytes:
            with st.spinner("Analyzing Audio modality..."):
                active_inputs["audio"] = audio_engine.analyze_audio(a_bytes)
        if t_text:
            with st.spinner("Analyzing Text modality..."):
                active_inputs["text"] = text_engine.analyze_text(t_text)

        if not active_inputs:
            st.error("Please provide at least ONE modality input to execute fusion.")
        else:
            with st.spinner("Computing Quality-Weighted Late Fusion & Cross-Modal Conflict Check..."):
                fusion_result = fusion_engine.fuse(
                    active_inputs,
                    custom_weights={"vision": w_v, "audio": w_a, "text": w_t}
                )

                # Persist result to SQLite
                add_prediction(
                    modality="multimodal",
                    dominant_emotion=fusion_result["fused_emotion"],
                    confidence=fusion_result["confidence"],
                    probabilities=fusion_result["probabilities"],
                    metadata={
                        "active_modalities": fusion_result["active_modalities"],
                        "conflict_detected": fusion_result["conflict_detected"],
                        "entropy": fusion_result["entropy"],
                        "normalized_weights": fusion_result["normalized_weights"]
                    },
                    notes=fusion_result["explanation"]
                )

            st.markdown("---")
            st.subheader("3. Multimodal Consensus Results")

            res_c1, res_c2 = st.columns([1, 1])

            with res_c1:
                emoji = EMOTION_EMOJIS.get(fusion_result["fused_emotion"], "🌟")
                st.markdown(f"## {emoji} Fused Emotion: **{fusion_result['fused_emotion'].upper()}**")
                st.metric("Fused Confidence", f"{fusion_result['confidence'] * 100:.1f}%")
                st.metric("Uncertainty (Entropy)", f"{fusion_result['entropy']:.3f} bits")

                if fusion_result["is_abstain"]:
                    st.warning("⚠️ **System Abstention:** Maximum confidence is low or uncertainty is high. The platform abstains from issuing a categorical classification.")

                if fusion_result["conflict_detected"]:
                    st.markdown(f"""
                    <div class="conflict-alert">
                        <strong>⚠️ Cross-Modal Conflict Detected!</strong><br>
                        {fusion_result['conflict_details']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="consensus-badge">
                        <strong>✅ Modal Concordance:</strong> Active modalities are in emotional alignment.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"**Audit Trail:** {fusion_result['explanation']}")

            with res_c2:
                plot_probabilities(fusion_result["probabilities"], "Fused Emotion Distribution")

            # Per-modality breakdown comparison
            st.subheader("4. Modality Breakdown & Effective Weights")
            breakdown_cols = st.columns(len(fusion_result["active_modalities"]))
            for idx, mod in enumerate(fusion_result["active_modalities"]):
                mod_data = active_inputs[mod]
                eff_w = fusion_result["normalized_weights"][mod]
                with breakdown_cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{mod.capitalize()} Modality</h4>
                        <p>Dominant: <strong>{mod_data['dominant_emotion'].upper()}</strong></p>
                        <p>Confidence: <strong>{mod_data['confidence']*100:.1f}%</strong></p>
                        <p>Effective Weight: <strong>{eff_w*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)


# ===========================================================================
# 6. PREDICTION ANALYTICS & HISTORY
# ===========================================================================
elif navigation_mode == "📊 Prediction Analytics":
    st.title("📊 Prediction Analytics & History")
    st.caption("Local SQLite Persistence • Historical Auditing • Emotion Distribution Trends")

    filter_modality = st.selectbox("Filter by Modality", ["All", "multimodal", "vision", "audio", "text"])
    modality_param = None if filter_modality == "All" else filter_modality

    history_records = get_history(limit=50, modality=modality_param)
    distribution = get_emotion_distribution(modality=modality_param)

    col_stat1, col_stat2 = st.columns([1, 1])

    with col_stat1:
        st.subheader("Historical Emotion Breakdown")
        if distribution:
            df_dist = pd.DataFrame(list(distribution.items()), columns=["Emotion", "Count"])
            st.bar_chart(df_dist.set_index("Emotion"))
        else:
            st.info("No prediction history recorded yet. Run an analysis above to populate records.")

    with col_stat2:
        st.subheader("Database Management")
        st.write(f"Total Records in Filtered View: **{len(history_records)}**")
        st.write("Privacy Safeguard: The platform stores only prediction metadata and confidence scores. No raw user images or voice audio are saved.")

        if st.button("🗑️ Clear Prediction History", type="secondary"):
            deleted = clear_history()
            st.success(f"Cleared {deleted} record(s) from database.")
            st.rerun()

    st.markdown("---")
    st.subheader("Recent Prediction Log")
    if history_records:
        table_data = []
        for r in history_records:
            table_data.append({
                "ID": r["id"],
                "Timestamp (UTC)": r["timestamp"][:19],
                "Modality": r["modality"].capitalize(),
                "Dominant Emotion": r["dominant_emotion"].capitalize(),
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Audit Notes": r["notes"] or "N/A"
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.write("Log is currently empty.")


# ===========================================================================
# 7. MODEL EVALUATION & BENCHMARKS
# ===========================================================================
elif navigation_mode == "🔬 Model Evaluation & Benchmarks":
    st.title("🔬 Model Evaluation & Benchmarking Explorer")
    st.caption("Rigorous Empirical Testing • Accuracy • Macro/Weighted F1 • Latency Profiling")

    report_path = PROJECT_ROOT / "evaluation_report.json"

    if st.button("▶️ Execute Benchmark Suite Now", type="primary"):
        with st.spinner("Executing Automated Benchmark Evaluation..."):
            from scripts.evaluate import main as run_eval_main
            run_eval_main()
            st.success("Evaluation complete! Report updated.")

    if report_path.exists():
        with open(report_path, "r") as f:
            report_data = json.load(f)

        tabs = st.tabs(["📝 Text (BERT)", "🎙️ Audio (SER)", "📷 Vision (MobileNetV2)"])

        for tab, (mod_key, mod_title) in zip(tabs, [("text", "BERT Sequence Classification"), ("audio", "Acoustic SER"), ("vision", "MobileNetV2 Vision")]):
            with tab:
                if mod_key in report_data:
                    info = report_data[mod_key]
                    metrics = info["metrics"]
                    lat = info["latency"]

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                    c2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
                    c3.metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")
                    c4.metric("Mean Latency", f"{lat['mean_ms']} ms", f"Median: {lat['median_ms']} ms")

                    st.markdown("#### Per-Class Performance")
                    df_per = pd.DataFrame.from_dict(metrics["per_class"], orient="index")
                    st.dataframe(df_per, use_container_width=True)

                    st.markdown("#### Confusion Matrix")
                    df_cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
                    st.dataframe(df_cm, use_container_width=True)
                else:
                    st.info(f"No benchmark data recorded for {mod_title}.")
    else:
        st.info("Evaluation report not found. Click 'Execute Benchmark Suite Now' to generate it.")


# ===========================================================================
# 8. ARCHITECTURE & MODEL REGISTRY
# ===========================================================================
elif navigation_mode == "⚙️ Architecture & Model Registry":
    st.title("⚙️ Architecture, Model Registry & Ethical Governance")
    st.markdown("### Technical Specifications and Provenance")

    st.markdown("""
    | Modality | Architecture / Model | Preprocessing | Input Format | Primary Labels | License |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Vision** | MobileNetV2 + Dense Head | Haar Face Crop, Resize 96x96, [0, 1] RGB | Image Array `(1, 96, 96, 3)` | 7 Ekman emotions | Apache 2.0 / Pretrained Weights |
    | **Audio** | Acoustic Prosodic & MFCC Classifier | 16kHz mono, silence filter, duration trim | 49-dim acoustic vector + Mel-spec | 7 Ekman emotions | Open Academic Research |
    | **Text** | Fine-Tuned BERT Sequence Classifier | Tokenization, truncation (max 128), lower | Token IDs & Attention Masks | 6 fine-grained mapped to Ekman | Apache 2.0 (Hugging Face) |
    | **Fusion**| Dynamic Quality-Weighted Late Fusion | Canonical 7-emotion mapping, entropy calculation | Modality Probabilities Dicts | 7 Canonical Emotions + Abstain | MIT Open Source |
    """)

    st.markdown("---")
    st.subheader("Ethical AI & Diagnostic Boundaries")
    st.write("""
    1. **Expression ≠ Internal Emotional State:** Human emotions are complex, nuanced, and culturally contextual. A smile does not prove happiness.
    2. **Non-Diagnostic Scope:** This software is designed strictly for research, user-experience analysis, and educational demonstration. It must not be deployed for criminal justice, employment decisions, or psychological diagnoses.
    3. **Privacy by Design:** User uploads are held in memory during the inference transaction and discarded immediately afterwards. Only aggregated prediction metadata is persisted in SQLite.
    """)
