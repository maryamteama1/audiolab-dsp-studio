import io
import time
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.fft as fft
import soundfile as sf
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Check for advanced spectral noise reduction
try:
    import noisereduce as nr
    HAS_NOISE_REDUCE = True
except ImportError:
    HAS_NOISE_REDUCE = False

st.set_page_config(
    page_title="AudioLab DSP Studio - Advanced",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Global Theme & Typography */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.2rem;
        color: #38BDF8;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #94A3B8;
        margin-bottom: 25px;
    }
    
    /* Modern Cards */
    .dsp-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 15px;
    }
    
    /* Metric Styling Overrides */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION

if "original_audio" not in st.session_state:
    st.session_state.original_audio = None
if "original_sr" not in st.session_state:
    st.session_state.original_sr = None
if "processed_audio" not in st.session_state:
    st.session_state.processed_audio = None
if "processed_sr" not in st.session_state:
    st.session_state.processed_sr = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = []
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------------------------------------------------
# AUDIO DECODER & HELPER FUNCTIONS

import tempfile
import os

def load_audio_file(uploaded_file):
    try:
        bytes_data = uploaded_file.read()
        suffix = os.path.splitext(uploaded_file.name)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(bytes_data)
            tmp_path = tmp_file.name

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=False)
            if y.ndim == 2:
                y = y.T  # (samples, channels)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")

def get_audio_metrics(y, sr):
    if y is None or len(y) == 0:
        return {}
    
    y_mono = np.mean(y, axis=1) if y.ndim > 1 else y

    peak = float(np.max(np.abs(y_mono)))
    rms = float(np.sqrt(np.mean(y_mono**2)))
    duration = float(len(y_mono) / sr)
    channels = 1 if y.ndim == 1 else y.shape[1]
    samples_per_channel = len(y_mono)
    energy = float(np.sum(y_mono**2))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(y_mono)))))
    
    return {
        "Duration (s)": round(duration, 3),
        "Sample Rate (Hz)": sr,
        "Channels": channels,
        "Total Samples": samples_per_channel,
        "Peak Amplitude": round(peak, 4),
        "RMS Amplitude": round(rms, 4),
        "Signal Energy": round(energy, 2),
        "Zero Crossing Rate": round(zcr, 4)
    }

def audio_to_bytes(y, sr):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='WAV', subtype='FLOAT')
    buffer.seek(0)
    return buffer.read()

# -----------------------------------------------------------------------------
# ADVANCED DSP OPERATIONS

def op_gain(y, sr, gain_db=0.0):
    linear_gain = 10.0 ** (gain_db / 20.0)
    return y * linear_gain, sr

def op_normalize(y, sr, target_peak=0.95):
    peak = np.max(np.abs(y))
    if peak == 0:
        return y, sr
    return y * (target_peak / peak), sr

def op_clipping(y, sr, threshold=0.8):
    return np.clip(y, -threshold, threshold), sr

def op_fade(y, sr, fade_in_ms=50.0, fade_out_ms=50.0):
    y_out = y.copy()
    n_samples = len(y)
    fi_samples = int((fade_in_ms / 1000.0) * sr)
    fo_samples = int((fade_out_ms / 1000.0) * sr)
    
    if fi_samples > 0 and n_samples > fi_samples:
        fade_in = np.linspace(0.0, 1.0, fi_samples)
        if y_out.ndim > 1:
            y_out[:fi_samples, :] *= fade_in[:, np.newaxis]
        else:
            y_out[:fi_samples] *= fade_in
            
    if fo_samples > 0 and n_samples > fo_samples:
        fade_out = np.linspace(1.0, 0.0, fo_samples)
        if y_out.ndim > 1:
            y_out[-fo_samples:, :] *= fade_out[:, np.newaxis]
        else:
            y_out[-fo_samples:] *= fade_out
            
    return y_out, sr

def op_filter(y, sr, filter_type="Low-pass", cutoff=1000.0, cutoff_high=3000.0, order=4):
    nyq = 0.5 * sr
    if filter_type == "Low-pass":
        normal_cutoff = min(cutoff / nyq, 0.99)
        sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
    elif filter_type == "High-pass":
        normal_cutoff = max(cutoff / nyq, 0.01)
        sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
    elif filter_type == "Band-pass":
        low = np.clip(cutoff / nyq, 0.001, 0.99)
        high = np.clip(cutoff_high / nyq, low + 0.001, 0.99)
        sos = signal.butter(order, [low, high], btype='band', output='sos')
    elif filter_type == "Band-stop":
        low = np.clip(cutoff / nyq, 0.001, 0.99)
        high = np.clip(cutoff_high / nyq, low + 0.001, 0.99)
        sos = signal.butter(order, [low, high], btype='stop', output='sos')
    else:
        return y, sr

    if y.ndim > 1:
        y_out = np.empty_like(y)
        for ch in range(y.shape[1]):
            y_out[:, ch] = signal.sosfiltfilt(sos, y[:, ch])
    else:
        y_out = signal.sosfiltfilt(sos, y)
    return y_out, sr

def op_denoise(y, sr, method="Spectral Gating (Stationary)", prop_decrease=0.8, window_size=5, stationary=True, auto_gain=True):
    y_work = y.copy()
    
    if method == "Wiener Adaptive Filter":
        # SciPy Statistical Wiener Filter
        if y_work.ndim > 1:
            y_out = np.empty_like(y_work)
            for ch in range(y_work.shape[1]):
                y_out[:, ch] = signal.wiener(y_work[:, ch], mysize=window_size)
        else:
            y_out = signal.wiener(y_work, mysize=window_size)

    elif method == "Moving Average":
        kernel = np.ones(window_size) / window_size
        if y_work.ndim > 1:
            y_out = np.empty_like(y_work)
            for ch in range(y_work.shape[1]):
                y_out[:, ch] = np.convolve(y_work[:, ch], kernel, mode='same')
        else:
            y_out = np.convolve(y_work, kernel, mode='same')

    elif method == "Median":
        w = window_size if window_size % 2 == 1 else window_size + 1
        if y_work.ndim > 1:
            y_out = np.empty_like(y_work)
            for ch in range(y_work.shape[1]):
                y_out[:, ch] = signal.medfilt(y_work[:, ch], kernel_size=w)
        else:
            y_out = signal.medfilt(y_work, kernel_size=w)

    elif "Spectral Gating" in method and HAS_NOISE_REDUCE:
        is_stat = "Stationary" in method
        if y_work.ndim > 1:
            y_out = nr.reduce_noise(y=y_work.T, sr=sr, prop_decrease=prop_decrease, stationary=is_stat).T
        else:
            y_out = nr.reduce_noise(y=y_work, sr=sr, prop_decrease=prop_decrease, stationary=is_stat)
    else:
        y_out = y_work

    # Automatic Gain Compensation to fix muffled/quiet audio!
    if auto_gain:
        orig_peak = np.max(np.abs(y))
        proc_peak = np.max(np.abs(y_out))
        if proc_peak > 0 and orig_peak > 0:
            scale_factor = min(orig_peak / proc_peak, 2.5) # Prevent extreme boost
            y_out = y_out * scale_factor

    return y_out, sr

def op_resample(y, sr, target_sr=16000):
    if sr == target_sr:
        return y, sr
    g = np.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    
    if y.ndim > 1:
        y_out = np.empty((int(len(y) * target_sr / sr), y.shape[1]), dtype=y.dtype)
        for ch in range(y.shape[1]):
            y_out[:, ch] = signal.resample_poly(y[:, ch], up, down)
    else:
        y_out = signal.resample_poly(y, up, down)
    return y_out, target_sr

def op_reverse(y, sr):
    return y[::-1], sr

def op_dc_removal(y, sr):
    if y.ndim > 1:
        y_out = y - np.mean(y, axis=0, keepdims=True)
    else:
        y_out = y - np.mean(y)
    return y_out, sr

def op_mono_stereo(y, sr, mode="To Mono"):
    if mode == "To Mono" and y.ndim > 1:
        return np.mean(y, axis=1), sr
    elif mode == "To Stereo" and y.ndim == 1:
        return np.column_stack((y, y)), sr
    return y, sr

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ AudioLab Studio Pro")
st.sidebar.markdown("Advanced DSP & Radio Speech Denoising")
st.sidebar.markdown("---")

nav_selection = st.sidebar.radio(
    "Navigation Workspace",
    ["1. Upload & Metadata", "2. Processing Pipeline", "3. Audio Comparison", "4. Signal Analysis", "5. Presets & Guide"]
)

st.sidebar.markdown("---")
if st.session_state.file_name:
    st.sidebar.success(f"Active File:\n**{st.session_state.file_name}**")
    if st.sidebar.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.original_audio = None
        st.session_state.original_sr = None
        st.session_state.processed_audio = None
        st.session_state.processed_sr = None
        st.session_state.file_name = None
        st.session_state.pipeline = []
        st.session_state.history = []
        st.rerun()
else:
    st.sidebar.info("No audio loaded.")

# -----------------------------------------------------------------------------
# MAIN WORKSPACE IMPLEMENTATION
# -----------------------------------------------------------------------------

if nav_selection == "1. Upload & Metadata":
    st.markdown('<p class="main-header">Audio Upload & Initialization</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload professional multi-format audio files (WAV, MP3, FLAC, OGG, AAC, AIFF).</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drop audio file here", type=["wav", "mp3", "flac", "ogg", "aac", "aiff", "m4a"])
    
    if uploaded_file is not None:
        if st.session_state.file_name != uploaded_file.name:
            with st.spinner("Decoding audio stream and calculating metrics..."):
                try:
                    y, sr = load_audio_file(uploaded_file)
                    st.session_state.original_audio = y
                    st.session_state.original_sr = sr
                    st.session_state.processed_audio = y.copy()
                    st.session_state.processed_sr = sr
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.pipeline = []
                    st.session_state.history = ["Initial Upload"]
                    st.success("Audio loaded successfully!")
                except Exception as e:
                    st.error(f"{e}")
                    st.info("💡 Note: For full `.aac` and `.m4a` decoding support, ensure **FFmpeg** is installed.")
                    st.stop()

    if st.session_state.original_audio is not None:
        st.markdown("### 📊 Audio Metrics & Metadata")
        metrics = get_audio_metrics(st.session_state.original_audio, st.session_state.original_sr)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Duration", f"{metrics['Duration (s)']} s")
            st.metric("Sample Rate", f"{metrics['Sample Rate (Hz)']} Hz")
        with c2:
            st.metric("Channels", f"{metrics['Channels']}")
            st.metric("Total Samples", f"{metrics['Total Samples']:,}")
        with c3:
            st.metric("Peak Amplitude", f"{metrics['Peak Amplitude']}")
            st.metric("RMS Amplitude", f"{metrics['RMS Amplitude']}")
        with c4:
            st.metric("Signal Energy", f"{metrics['Signal Energy']:.1f}")
            st.metric("Zero Crossing Rate", f"{metrics['Zero Crossing Rate']}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎧 Original Audio Player")
        st.audio(audio_to_bytes(st.session_state.original_audio, st.session_state.original_sr), format='audio/wav')
    else:
        st.info("👈 Upload an audio file using the uploader above to get started.")

elif nav_selection == "2. Processing Pipeline":
    st.markdown('<p class="main-header">Composable DSP Pipeline</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chain multiple operations sequentially for custom engineering workflows.</p>', unsafe_allow_html=True)
    
    if st.session_state.original_audio is None:
        st.warning("Please upload an audio file in Section 1 first.")
    else:
        st.markdown("### ➕ Add DSP Operation")
        
        # Immediate UI Update Fix (No need for form execution bug)
        op_type = st.selectbox(
            "Select Operation Type",
            [
                "Noise Reduction",
                "Digital Filter",
                "Peak Normalization",
                "Gain Adjustment",
                "Amplitude Clipping",
                "Fade Envelopes",
                "Resampling",
                "Time Reversal",
                "DC Offset Removal",
                "Mono / Stereo Conversion"
            ]
        )
        
        # Form created with key bound to operation type for automatic UI refresh
        with st.form(key=f"dsp_op_form_{op_type}"):
            params = {}
            st.markdown(f"**Parameters for `{op_type}`:**")
            
            if op_type == "Noise Reduction":
                methods = ["Wiener Adaptive Filter", "Moving Average", "Median"]
                if HAS_NOISE_REDUCE:
                    methods.insert(0, "Spectral Gating (Stationary)")
                    methods.insert(1, "Spectral Gating (Non-Stationary)")
                
                params["method"] = st.selectbox("Noise Reduction Method", methods)
                params["prop_decrease"] = st.slider("Reduction Power (Prop Decrease)", 0.1, 1.0, 0.75, 0.05, 
                                                    help="Higher values clean more noise but can muffle sound. Lower values preserve speech.")
                params["window_size"] = st.slider("Filter Window Size", 3, 31, 7, step=2)
                params["auto_gain"] = st.checkbox("Auto-Gain Compensation", value=True, 
                                                  help="Automatically restores lost loudness after noise suppression.")

            elif op_type == "Digital Filter":
                params["filter_type"] = st.selectbox("Type", ["Low-pass", "High-pass", "Band-pass", "Band-stop"])
                params["cutoff"] = st.number_input("Cutoff / Low Freq (Hz)", 20.0, 20000.0, 300.0)
                if params["filter_type"] in ["Band-pass", "Band-stop"]:
                    params["cutoff_high"] = st.number_input("High Cut Freq (Hz)", 100.0, 20000.0, 3400.0)
                params["order"] = st.slider("Filter Order (Butterworth)", 1, 8, 4)

            elif op_type == "Gain Adjustment":
                params["gain_db"] = st.slider("Gain (dB)", -30.0, 30.0, 3.0, 0.5)

            elif op_type == "Peak Normalization":
                params["target_peak"] = st.slider("Target Peak Amplitude", 0.1, 1.0, 0.95, 0.01)

            elif op_type == "Amplitude Clipping":
                params["threshold"] = st.slider("Clipping Threshold", 0.1, 1.0, 0.8, 0.05)

            elif op_type == "Fade Envelopes":
                params["fade_in_ms"] = st.number_input("Fade In (ms)", 0.0, 5000.0, 100.0)
                params["fade_out_ms"] = st.number_input("Fade Out (ms)", 0.0, 5000.0, 100.0)

            elif op_type == "Resampling":
                params["target_sr"] = st.selectbox("Target Sample Rate (Hz)", [8000, 11025, 16000, 22050, 32000, 44100, 48000], index=2)

            elif op_type == "Mono / Stereo Conversion":
                params["mode"] = st.selectbox("Mode", ["To Mono", "To Stereo"])

            submitted = st.form_submit_button("➕ Add to Pipeline Stack", use_container_width=True)
            if submitted:
                st.session_state.pipeline.append({"type": op_type, "params": params})
                st.success(f"Successfully added '{op_type}' to active stack.")
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Active Pipeline Queue")
        if len(st.session_state.pipeline) == 0:
            st.info("Pipeline queue is empty. Add operations using the controls above.")
        else:
            for idx, step in enumerate(st.session_state.pipeline):
                c1, c2, c3 = st.columns([0.5, 3.5, 1])
                with c1:
                    st.markdown(f"**#{idx+1}**")
                with c2:
                    st.markdown(f"`{step['type']}` : {step['params']}")
                with c3:
                    if st.button("Remove", key=f"rem_{idx}", use_container_width=True):
                        st.session_state.pipeline.pop(idx)
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("🚀 Execute Pipeline Stack", type="primary", use_container_width=True):
                    with st.spinner("Executing sequence in real-time..."):
                        curr_y = st.session_state.original_audio.copy()
                        curr_sr = st.session_state.original_sr
                        history_log = ["Original Upload"]
                        
                        for step in st.session_state.pipeline:
                            t = step["type"]
                            p = step["params"]
                            if t == "Gain Adjustment":
                                curr_y, curr_sr = op_gain(curr_y, curr_sr, **p)
                            elif t == "Peak Normalization":
                                curr_y, curr_sr = op_normalize(curr_y, curr_sr, **p)
                            elif t == "Amplitude Clipping":
                                curr_y, curr_sr = op_clipping(curr_y, curr_sr, **p)
                            elif t == "Fade Envelopes":
                                curr_y, curr_sr = op_fade(curr_y, curr_sr, **p)
                            elif t == "Digital Filter":
                                curr_y, curr_sr = op_filter(curr_y, curr_sr, **p)
                            elif t == "Noise Reduction":
                                curr_y, curr_sr = op_denoise(curr_y, curr_sr, **p)
                            elif t == "Resampling":
                                curr_y, curr_sr = op_resample(curr_y, curr_sr, **p)
                            elif t == "Time Reversal":
                                curr_y, curr_sr = op_reverse(curr_y, curr_sr)
                            elif t == "DC Offset Removal":
                                curr_y, curr_sr = op_dc_removal(curr_y, curr_sr)
                            elif t == "Mono / Stereo Conversion":
                                curr_y, curr_sr = op_mono_stereo(curr_y, curr_sr, **p)
                            history_log.append(t)
                        
                        st.session_state.processed_audio = curr_y
                        st.session_state.processed_sr = curr_sr
                        st.session_state.history = history_log
                        st.success("Pipeline executed successfully!")
            with col_act2:
                if st.button("🔄 Clear Stack & Reset Audio", use_container_width=True):
                    st.session_state.pipeline = []
                    st.session_state.processed_audio = st.session_state.original_audio.copy()
                    st.session_state.processed_sr = st.session_state.original_sr
                    st.rerun()

elif nav_selection == "3. Audio Comparison":
    st.markdown('<p class="main-header">Before & After Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluate original versus processed audio outputs side by side.</p>', unsafe_allow_html=True)
    
    if st.session_state.original_audio is None:
        st.warning("Please upload an audio file first.")
    else:
        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.markdown("### 🎵 Original Stream")
            st.audio(audio_to_bytes(st.session_state.original_audio, st.session_state.original_sr), format='audio/wav')
            orig_m = get_audio_metrics(st.session_state.original_audio, st.session_state.original_sr)
            for k, v in orig_m.items():
                st.write(f"- **{k}**: `{v}`")
                
        with col_proc:
            st.markdown("### 🎶 Processed Stream")
            st.audio(audio_to_bytes(st.session_state.processed_audio, st.session_state.processed_sr), format='audio/wav')
            proc_m = get_audio_metrics(st.session_state.processed_audio, st.session_state.processed_sr)
            for k, v in proc_m.items():
                st.write(f"- **{k}**: `{v}`")
                
        st.markdown("---")
        st.markdown("### 💾 Export Processed Audio")
        wav_bytes = audio_to_bytes(st.session_state.processed_audio, st.session_state.processed_sr)
        st.download_button(
            label="Download Processed Audio (.WAV)",
            data=wav_bytes,
            file_name="audiolab_processed.wav",
            mime="audio/wav",
            use_container_width=True
        )

elif nav_selection == "4. Signal Analysis":
    st.markdown('<p class="main-header">Signal Analysis Workspace</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced inspection via Time-Domain, FFT Spectrum, Spectrogram, and Metrics.</p>', unsafe_allow_html=True)
    
    if st.session_state.original_audio is None:
        st.warning("Please upload an audio file first.")
    else:
        source_choice = st.radio("Target Signal", ["Original Signal", "Processed Signal"], horizontal=True)
        y_target = st.session_state.original_audio if source_choice == "Original Signal" else st.session_state.processed_audio
        sr_target = st.session_state.original_sr if source_choice == "Original Signal" else st.session_state.processed_sr
        y_plot = np.mean(y_target, axis=1) if y_target.ndim > 1 else y_target
        
        tab_time, tab_freq, tab_spec, tab_stats = st.tabs(["Time Domain", "Frequency Domain (FFT)", "Spectrogram", "Audio Descriptors"])
        
        with tab_time:
            time_axis = np.linspace(0, len(y_plot) / sr_target, num=len(y_plot))
            step_dec = max(1, len(y_plot) // 5000)
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=time_axis[::step_dec], y=y_plot[::step_dec],
                mode='lines', line=dict(color='#38BDF8', width=1)
            ))
            fig_time.update_layout(
                xaxis_title="Time (s)", yaxis_title="Amplitude",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F8FAFC'), margin=dict(l=20, r=20, t=20, b=20), height=380
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
        with tab_freq:
            n = len(y_plot)
            window = np.hanning(n)
            fft_vals = fft.rfft(y_plot * window)
            fft_freqs = fft.rfftfreq(n, 1/sr_target)
            magnitude = np.abs(fft_vals) / n
            
            fig_freq = go.Figure()
            fig_freq.add_trace(go.Scatter(
                x=fft_freqs, y=magnitude,
                mode='lines', line=dict(color='#38BDF8', width=1.5)
            ))
            fig_freq.update_layout(
                xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F8FAFC'), margin=dict(l=20, r=20, t=20, b=20), height=380
            )
            st.plotly_chart(fig_freq, use_container_width=True)
            
        with tab_spec:
            D = librosa.stft(y_plot, n_fft=2048, hop_length=512)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            fig_spec = go.Figure(data=go.Heatmap(
                z=S_db,
                x=np.linspace(0, len(y_plot)/sr_target, S_db.shape[1]),
                y=librosa.fft_frequencies(sr=sr_target, n_fft=2048),
                colorscale='Turbo'
            ))
            fig_spec.update_layout(
                xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F8FAFC'), margin=dict(l=20, r=20, t=20, b=20), height=400
            )
            st.plotly_chart(fig_spec, use_container_width=True)
            
        with tab_stats:
            spectral_centroids = librosa.feature.spectral_centroid(y=y_plot, sr=sr_target)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_plot, sr=sr_target)[0]
            spectral_flatness = librosa.feature.spectral_flatness(y=y_plot)[0]
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Mean Centroid", f"{np.mean(spectral_centroids):.2f} Hz")
                st.metric("Mean Bandwidth", f"{np.mean(spectral_bandwidth):.2f} Hz")
            with col_s2:
                st.metric("Spectral Flatness", f"{np.mean(spectral_flatness):.4f}")
                st.metric("RMS Amplitude", f"{np.sqrt(np.mean(y_plot**2)):.4f}")
            with col_s3:
                st.metric("Peak Amplitude", f"{np.max(np.abs(y_plot)):.4f}")
                st.metric("Zero Crossing Rate", f"{np.mean(np.abs(np.diff(np.signbit(y_plot)))):.4f}")

elif nav_selection == "5. Presets & Guide":
    st.markdown('<p class="main-header">Presets & DSP Guide</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Load engineering presets and study fundamental concepts.</p>', unsafe_allow_html=True)
    
    st.markdown("### ⚡ Quick Presets")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("Load: Voice Clarity (HPF + Norm)", use_container_width=True):
            if st.session_state.original_audio is not None:
                st.session_state.pipeline = [
                    {"type": "Digital Filter", "params": {"filter_type": "High-pass", "cutoff": 80.0, "order": 4}},
                    {"type": "Peak Normalization", "params": {"target_peak": 0.95}}
                ]
                st.success("Loaded Voice Clarity preset!")
            else:
                st.warning("Upload audio first.")
    with col_p2:
        if st.button("Load: Speech Prep (16kHz + LPF)", use_container_width=True):
            if st.session_state.original_audio is not None:
                st.session_state.pipeline = [
                    {"type": "Digital Filter", "params": {"filter_type": "Low-pass", "cutoff": 7500.0, "order": 4}},
                    {"type": "Resampling", "params": {"target_sr": 16000}}
                ]
                st.success("Loaded Speech Prep preset!")
            else:
                st.warning("Upload audio first.")
    with col_p3:
        if st.button("Load: Heavy Radio Denoise (Pro)", use_container_width=True):
            if st.session_state.original_audio is not None:
                st.session_state.pipeline = [
                    {"type": "Digital Filter", "params": {"filter_type": "Band-pass", "cutoff": 300.0, "cutoff_high": 3400.0, "order": 4}},
                    {"type": "Noise Reduction", "params": {"method": "Wiener Adaptive Filter", "prop_decrease": 0.7, "window_size": 7, "auto_gain": True}},
                    {"type": "Peak Normalization", "params": {"target_peak": 0.95}}
                ]
                st.success("Loaded Heavy Radio Denoise preset!")
            else:
                st.warning("Upload audio first.")

    st.markdown("---")
    st.markdown("### 📚 Educational Engineering Notes")
    with st.expander("Zero-Phase Filtering"):
        st.write("Using `sosfiltfilt` allows bidirectional filtering, preventing phase distortion entirely.")
    with st.expander("Polyphase Resampling"):
        st.write("Proper anti-aliasing interpolation/decimation prevents high-frequency folding artifacts.")
    with st.expander("Wiener Adaptive Filter"):
        st.write("Calculates minimum mean square error (MMSE) between noisy signal and local variance to dynamically attenuate background noise without muffling voice.")