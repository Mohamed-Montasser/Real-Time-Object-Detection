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

import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import sys
from pathlib import Path

# ONNX-specific imports
import onnxruntime as ort
from ultralytics import YOLO  # Still needed for some utilities

st.write(f"Python: {sys.version}")

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
MODEL_PATH = "weights/best.onnx"
INPUT_SIZE = 640  # Standard YOLO input size

class ONNXModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
    def predict(self, image, conf_threshold=0.5):
        # Preprocess
        img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: img})
        
        # Post-process (simplified - you'll need to adapt this to your model's output format)
        # This part will vary based on how your ONNX model was exported
        # You may need to use YOLO's native postprocessing or implement your own
        return outputs

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"ONNX model file not found at {MODEL_PATH}")
            return None
            
        file_size = os.path.getsize(MODEL_PATH)/(1024*1024)  # Size in MB
        if file_size < 10:  # Adjust based on your expected model size
            st.error(f"Model file seems too small ({file_size:.2f} MB). Expected at least 10MB.")
            return None
            
        # Verify file content
        with open(MODEL_PATH, 'rb') as f:
            header = f.read(10)
            if header.startswith(b'<') or b'html' in header.lower():
                st.error("File appears to be HTML, not a model file")
                return None
                
        # Load ONNX model
        model = ONNXModel(MODEL_PATH)
        
        # Quick test prediction
        try:
            dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            model.predict(dummy)
        except Exception as e:
            st.error(f"Model test failed: {str(e)}")
            return None
            
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def process_frame(_model, frame, conf_threshold):
    try:
        # Preprocess frame
        frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        
        # Get predictions
        outputs = _model.predict(frame_resized, conf_threshold)
        
        # This is a placeholder - you'll need to implement proper visualization
        # based on your ONNX model's output format
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, "ONNX Model Working", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), outputs
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return frame, None

def display_detections(results):
    """Placeholder - implement based on your ONNX output format"""
    with st.sidebar.expander("📊 Detection Stats", expanded=True):
        st.warning("Detection display needs implementation based on ONNX output format")

def main():
    st.title("🚦 BDD10K Traffic Object Detection (ONNX)")
    st.caption("Detect vehicles, pedestrians, and traffic elements in images/videos")
    
    with st.sidebar:
        st.header("Settings")
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

    model = load_model()
    if not model:
        return

    uploaded_file = st.file_uploader(
        "Upload media", 
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        accept_multiple_files=False,
        help="Supports images and videos"
    )

    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        if file_ext in ["jpg", "jpeg", "png"]:
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
                    
                    if frame_count % 5 == 0:
                        st_frame.image(processed_frame, channels="RGB")
                    
                    progress = frame_count / total_frames
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Processing: {frame_count}/{total_frames} frames")
                    frame_count += 1
                
                cap.release()
                os.unlink(tfile.name)
                
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
