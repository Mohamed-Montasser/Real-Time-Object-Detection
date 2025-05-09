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

# Import required libraries
import numpy as np
import tempfile
import os
import time
import sys
import io
import requests
import logging
from PIL import Image
import cv2
import torch
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Display environment info
st.write(f"Python: {sys.version}")
st.write(f"PyTorch: {torch.__version__}")
st.write(f"OpenCV: {cv2.__version__}")

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
        "url": "https://drive.google.com/uc?id=10h9qk50tdkVrBQ2czqPF3rvsgXuDMpDJ",
        "path": "best.pt",  # Local save path
        "id": "10h9qk50tdkVrBQ2czqPF3rvsgXuDMpDJ"
    },
    "ONNX (.onnx)": {
        "url": "https://drive.google.com/uc?id=13RtUuLQa4HdK2w1qUtFm8RRA0WafkSXW",
        "path": "best.onnx",  # Local save path
        "id": "13RtUuLQa4HdK2w1qUtFm8RRA0WafkSXW"
    }
}

def download_file_from_google_drive(file_id, destination):
    """
    More robust Google Drive downloader with direct API approach
    """
    try:
        # Define session and headers
        session = requests.Session()
        
        # Step 1: Get confirmation token
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        response = session.get(url, stream=True)
        
        # Check if we have a small file without confirmation
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # Step 2: Get the actual file with token if needed
        if token:
            url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={token}"
            response = session.get(url, stream=True)
        
        # Step 3: Save the file
        with open(destination, 'wb') as f:
            total_length = response.headers.get('content-length')
            dl = 0
            total_length = int(total_length) if total_length else None
            
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    dl += len(chunk)
                    f.write(chunk)
                    # Show download progress
                    if total_length is not None:
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s] %s%%" % ('=' * done, ' ' * (50-done), int(100 * dl / total_length)))
                        sys.stdout.flush()
        
        # Verify file isn't HTML
        with open(destination, 'rb') as f:
            header = f.read(20)
            if b'<html' in header.lower() or b'<!doctype html' in header.lower():
                os.remove(destination)
                return False
        
        return True
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False

@st.cache_resource(show_spinner=False)
def load_model(model_info):
    """Load or download model with improved error handling"""
    model_path = model_info["path"]
    file_id = model_info["id"]
    
    # Try 3 times to download
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Check if model exists
            if not os.path.exists(model_path):
                with st.spinner(f"Downloading model (attempt {attempt+1}/{max_attempts})..."):
                    success = download_file_from_google_drive(file_id, model_path)
                    if not success:
                        if attempt < max_attempts - 1:
                            st.warning(f"Download attempt {attempt+1} failed. Retrying...")
                            time.sleep(1)  # Wait before retry
                            continue
                        else:
                            st.error("Failed to download model after multiple attempts.")
                            st.info("""
                            ### Troubleshooting:
                            1. The Google Drive file might be restricted or unavailable
                            2. Try uploading your own model file using the uploader below
                            3. Or try using a pre-trained YOLOv8 model
                            """)
                            return None
            
            # Verify file size is reasonable for a model
            file_size = os.path.getsize(model_path)
            if file_size < 10000:  # Less than 10KB is suspicious for a model
                st.warning(f"Downloaded file seems too small ({file_size} bytes), might not be a valid model")
            
            # Try to load the model
            model = YOLO(model_path)
            st.success(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            if os.path.exists(model_path):
                os.remove(model_path)  # Remove potentially corrupt file
            if attempt < max_attempts - 1:
                st.warning(f"Retrying download (attempt {attempt+2}/{max_attempts})...")
            else:
                st.error("Failed to load model after multiple attempts.")
                return None

def load_pretrained_model():
    """Load a pretrained YOLOv8 model as fallback"""
    try:
        with st.spinner("Loading pre-trained YOLOv8 model..."):
            model = YOLO("yolov8n.pt")  # Use the smallest YOLOv8 model
            st.success("Pre-trained YOLOv8 model loaded successfully")
            return model
    except Exception as e:
        st.error(f"Error loading pre-trained model: {str(e)}")
        return None

def process_frame(_model, frame, conf_threshold):
    """Process a single frame with error handling"""
    try:
        results = _model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot(line_width=2, font_size=10)
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), results
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return frame, None

def display_detections(results):
    """Display detection results in sidebar"""
    if results:
        detections = []
        for box in results[0].boxes:
            cls_idx = int(box.cls)
            # For standard YOLOv8 model with COCO classes
            if hasattr(results[0], 'names') and results[0].names:
                label = results[0].names.get(cls_idx, f"Class {cls_idx}")
            # For BDD10K custom model
            elif cls_idx < len(CUSTOM_LABELS):
                label = CUSTOM_LABELS[cls_idx]
            else:
                label = f"Class {cls_idx}"  # Fallback label
                
            detections.append({
                "label": label,
                "confidence": float(box.conf)
            })
        
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
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        model_options = list(MODEL_PATHS.keys()) + ["Pre-trained YOLOv8"]
        model_type = st.radio(
            "Model Selection",
            model_options,
            key="model_format_selector"  # Unique key
        )
        
        # Alternative model upload option
        custom_model = st.file_uploader(
            "Or upload your own model file",
            type=["pt", "onnx"],
            help="Upload your own trained model if downloads aren't working"
        )
        
        conf_threshold = st.slider(
            "Confidence Threshold", 
            0.1, 0.9, 0.5, 0.05,
            help="Adjust to filter weak detections"
        )
        
        st.divider()
        st.info("""
        **Instructions:**
        1. Upload image/video
        2. Click 'Process'
        3. View results
        """)

    # Load model with fallbacks
    model = None
    
    if custom_model:
        # Save uploaded model to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(custom_model.name)[1]) as tmp_file:
            tmp_file.write(custom_model.getvalue())
            custom_model_path = tmp_file.name
        
        try:
            model = YOLO(custom_model_path)
            st.success(f"Custom model loaded: {custom_model.name}")
        except Exception as e:
            st.error(f"Error loading custom model: {e}")
    elif model_type == "Pre-trained YOLOv8":
        model = load_pretrained_model()
    else:
        model = load_model(MODEL_PATHS[model_type])
    
    if not model:
        st.warning("No working model available. Please try a pre-trained model or upload a custom model.")
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
                
                # Check if video opened successfully
                if not cap.isOpened():
                    st.error("Error opening video file!")
                    return
                
                st_frame = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000  # Fallback if unknown
                processed_frames = []
                
                # Process frames
                try:
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
                except Exception as e:
                    st.error(f"Error during video processing: {e}")
                finally:
                    cap.release()
                
                os.unlink(tfile.name)
                
                # Save processed video
                if processed_frames:
                    try:
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
                    except Exception as e:
                        st.error(f"Error saving video: {e}")

    # Show sample images if no file uploaded
    if not uploaded_file:
        st.info("Upload an image or video to begin object detection")
        # Could add sample images here in the future

if __name__ == "__main__":
    main()
