# 🚦 Object Detection Web App

![Redirect to the website]([https://via.placeholder.com/800x500.png?text=BDD10K+Object+Detection+Demo](https://real-time-object-detection-for-autonomous-vehicles.streamlit.app/))

A Streamlit web application for detecting traffic objects (cars, pedestrians, traffic signs, etc.) in images and videos using YOLOv8 models trained on the BDD10K dataset.

## 🚀 Features

- 🚗 Detect 10 classes of traffic objects (cars, buses, trucks, pedestrians, etc.)
- 📷 Process both images and videos
- ⚡ Two pre-trained model options (PyTorch and ONNX formats)
- 📤 Upload your own custom YOLOv8 models
- 📊 Real-time detection statistics and confidence metrics
- 🎨 Custom UI with gradient styling
- 🔍 Debug mode for troubleshooting

## ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/bdd10k-object-detection.git
cd bdd10k-object-detection

# Install dependencies
pip install -r requirements.txt
```

# 🖥️ Usage

In the sidebar:

- Choose a pre-trained model or upload your own
- Download the selected model if needed
- Click "Load Model" to initialize the detector
- Adjust the confidence threshold as needed

Upload an image or video file and click "Process"

---


