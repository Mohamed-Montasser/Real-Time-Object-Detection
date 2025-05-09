# MUST BE FIRST STREAMLIT COMMAND
import streamlit as st
try:
    st.set_page_config(
        page_title="BDD10K Object Detection",
        page_icon="🚗", 
        layout="wide"
    )
except Exception as e:
    st.error(f"Page config error: {e}")

# Import necessary libraries
import sys
import os
import time
import tempfile
import numpy as np
from PIL import Image
import torch

# Display version information
st.write(f"Python: {sys.version}")
st.write(f"PyTorch: {torch.__version__}")

# Install required packages if needed
try:
    from ultralytics import YOLO
    import ultralytics
    st.write(f"Ultralytics: {ultralytics.__version__}")
except ImportError:
    st.info("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO
    import ultralytics
    st.write(f"Ultralytics: {ultralytics.__version__}")

try:
    import cv2
except ImportError:
    st.info("Installing OpenCV...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

try:
    import gdown
except ImportError:
    st.info("Installing gdown...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

# Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        color: white;
    }
    .stButton>button {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
    }
    .stDownloadButton>button {
        background: linear-gradient(to right, #a1c4fd 0%, #c2e9fb 100%);
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CUSTOM_LABELS = ["car", "train", "motor", "person", "bus", "truck", "bike", 
                "rider", "traffic light", "traffic sign"]
MODEL_PATHS = {
    "PyTorch (.pt)": {
        "url": "https://drive.google.com/uc?export=download&id=10h9qk50tdkVrBQ2czqPF3rvsgXuDMpDJ",
        "path": os.path.join(os.getcwd(), "best.pt")  # Full path
    },
    "ONNX (.onnx)": {
        "url": "https://drive.google.com/uc?export=download&id=13RtUuLQa4HdK2w1qUtFm8RRA0WafkSXW",
        "path": os.path.join(os.getcwd(), "best.onnx")  # Full path
    }
}

# Ensure model directory exists
os.makedirs(os.getcwd(), exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_model(model_type):
    """Load the model with proper error handling and verification"""
    if model_type not in MODEL_PATHS:
        st.error(f"Unknown model type: {model_type}")
        return None
        
    model_info = MODEL_PATHS[model_type]
    model_path = model_info["path"]
    
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Download if file doesn't exist
        if not os.path.exists(model_path):
            with st.spinner(f"Downloading model from Google Drive..."):
                try:
                    gdown.download(model_info["url"], model_path, quiet=False, use_cookies=True)
                    st.success(f"Model downloaded to {model_path}")
                except Exception as download_error:
                    st.error(f"Download error: {download_error}")
                    return None
        
        # Verify file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
            
        # Check file size
        file_size = os.path.getsize(model_path)
        if file_size < 10000:  # Less than 10KB is probably not a valid model
            with open(model_path, 'rb') as f:
                header = f.read(100)
                if b'<!DOCTYPE html>' in header or b'<html' in header:
                    st.error("Downloaded file is HTML, not a model file. Drive link may require authentication.")
                    # Remove the invalid file
                    os.remove(model_path)
                    return None
        
        st.info(f"Loading model from {model_path}...")
        
        # Load model with explicit task
        model = YOLO(model_path, task='detect')
        st.success("Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def process_frame(model, frame, conf_threshold):
    """Process a single frame with error handling"""
    if model is None:
        return frame, None
        
    try:
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot(line_width=2, font_size=10)
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), results
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return frame, None

def display_detections(results):
    """Display detection results in sidebar"""
    if results and results[0].boxes:
        detections = []
        for box in results[0].boxes:
            class_idx = int(box.cls)
            if class_idx < len(CUSTOM_LABELS):
                detections.append({
                    "label": CUSTOM_LABELS[class_idx],
                    "confidence": float(box.conf)
                })
            else:
                st.warning(f"Unknown class index: {class_idx}")
        
        if detections:
            with st.sidebar.expander("📊 Detection Stats", expanded=True):
                st.subheader("Detected Objects")
                for det in sorted(detections, key=lambda x: x['confidence'], reverse=True):
                    st.progress(det['confidence'], 
                               text=f"{det['label']}: {det['confidence']:.2f}")
                
                st.subheader("Class Distribution")
                class_counts = {}
                for det in detections:
                    class_counts[det['label']] = class_counts.get(det['label'], 0) + 1
                
                for label, count in class_counts.items():
                    st.metric(label=label, value=count)

def main():
    st.title("🚦 BDD10K Traffic Object Detection")
    st.caption("Detect vehicles, pedestrians, and traffic elements in images/videos")
    
    # Show debug info
    if st.checkbox("Show Debug Info"):
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Files in directory: {os.listdir(os.getcwd())}")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        model_type = st.radio(
            "Model Format",
            list(MODEL_PATHS.keys()),
            index=0,
            key="model_format_selector"
        )
        
        # Try loading model with button to give more control
        if st.button("Load Selected Model"):
            with st.spinner("Loading model..."):
                model = load_model(model_type)
                if model:
                    st.session_state['model'] = model
                    st.success(f"Model loaded successfully!")
                else:
                    st.error("Failed to load model. Check errors above.")
        
        conf_threshold = st.slider(
            "Confidence Threshold", 
            0.1, 0.9, 0.5, 0.05,
            help="Adjust to filter weak detections"
        )
        
        st.divider()
        st.info("""
        **Instructions:**
        1. Select model format
        2. Click "Load Selected Model"
        3. Upload image/video
        4. Click 'Process'
        5. View results
        """)

    # Check if model is loaded
    model = st.session_state.get('model', None)
    if model is None:
        st.warning("Please load a model from the sidebar first.")
        return

    # File upload
    uploaded_file = st.file_uploader(
        "Upload media", 
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        accept_multiple_files=False,
        help="Supports images and videos"
    )

    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        if file_ext in ["jpg", "jpeg", "png"]:
            # Image processing
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            if st.button("✨ Process Image", type="primary"):
                with st.spinner("Detecting objects..."):
                    start_time = time.time()
                    frame = np.array(image)
                    annotated_frame, results = process_frame(model, frame, conf_threshold)
                    processing_time = time.time() - start_time
                    
                    with col2:
                        st.subheader("Processed Result")
                        st.image(annotated_frame, use_column_width=True)
                        st.caption(f"Processed in {processing_time:.2f} seconds")
                    
                    display_detections(results)
        
        elif file_ext in ["mp4", "avi", "mov"]:
            # Video processing
            st.subheader("Uploaded Video Preview")
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
            tfile.write(uploaded_file.read())
            
            video_bytes = open(tfile.name, 'rb').read()
            st.video(video_bytes)
            
            if st.button("🎥 Process Video", type="primary"):
                st.warning("Video processing may take time. Please wait...")
                cap = cv2.VideoCapture(tfile.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                st_frame = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processed_frames = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame, _ = process_frame(model, frame, conf_threshold)
                    processed_frames.append(processed_frame)
                    
                    # Display every nth frame to improve performance
                    if frame_count % 5 == 0:
                        st_frame.image(processed_frame, channels="RGB")
                    
                    progress = frame_count / total_frames
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Processing: {frame_count}/{total_frames} frames")
                    frame_count += 1
                
                cap.release()
                os.unlink(tfile.name)
                
                # Save processed video
                if processed_frames:
                    output_path = "processed_video.mp4"
                    height, width = processed_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    for frame in processed_frames:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    out.release()
                    
                    st.success("Processing complete!")
                    st.download_button(
                        label="Download Processed Video",
                        data=open(output_path, 'rb').read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

if __name__ == "__main__":
    main()
