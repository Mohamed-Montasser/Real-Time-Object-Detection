import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import gdown
# Set page config
st.set_page_config(
    page_title="BDD10K Object Detection",
    page_icon="🚗",
    layout="wide"
)

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
        "url": "https://drive.google.com/drive/folders/1p6d1QxxzmWfGFJjyij9TqEG3ZzVk3nvP?usp=sharing",
        "path": "best.pt"  # Local save path
    },
    "ONNX (.onnx)": {
        "url": "https://drive.google.com/drive/folders/1p6d1QxxzmWfGFJjyij9TqEG3ZzVk3nvP?usp=sharing",
        "path": "best.onnx"  # Local save path
    }
}

@st.cache_resource(show_spinner=False)
def load_model(model_info):
    """Download (if needed) and load model from Google Drive"""
    os.makedirs("weights", exist_ok=True)  # Create weights folder
    
    # Download if file doesn't exist
    if not os.path.exists(model_info["path"]):
        try:
            with st.spinner(f"Downloading model..."):
                gdown.download(model_info["url"], model_info["path"], quiet=False)
        except Exception as e:
            st.error(f"Download failed: {e}")
            return None
    
    # Load the model
    try:
        model = YOLO(model_info["path"])
        model.names = CUSTOM_LABELS
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
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
            detections.append({
                "label": CUSTOM_LABELS[int(box.cls)],
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
        model_type = st.radio(
            "Model Format",
            list(MODEL_PATHS.keys()),
            index=0
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

    # Load model
    model_type = st.sidebar.radio("Model Format", list(MODEL_PATHS.keys()))
    model = load_model(MODEL_PATHS[model_type])
    if not model:
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
