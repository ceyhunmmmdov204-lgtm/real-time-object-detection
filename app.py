import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from object_detector import ObjectDetector
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Real-Time Object Detection",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header"> Real-Time Object Detection</h1>', unsafe_allow_html=True)

# Sidebar - Configuration
with st.sidebar:
    st.markdown('<h3 class="sub-header"> Configuration</h3>', unsafe_allow_html=True)
    
    # Model selection
    model_type = st.selectbox(
        "Model Type",
        ["YOLOv8n (Small)", "YOLOv8s (Medium)", "YOLOv8m (Large)"]
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.0, 1.0, 0.5, 0.05
    )
    
    # NMS threshold
    iou_threshold = st.slider(
        "NMS Threshold", 0.0, 1.0, 0.45, 0.05
    )
    
    # Image size
    img_size = st.slider(
        "Image Size", 320, 1280, 640, step=32
    )
    
    # Object filtering
    st.markdown("### Objects to Detect")
    col1, col2 = st.columns(2)
    with col1:
        show_person = st.checkbox("Person", True)
        show_vehicle = st.checkbox("Vehicle", True)
    with col2:
        show_animal = st.checkbox("Animal", False)
        show_electronics = st.checkbox("Electronics", False)
    
    st.markdown("---")
    
    # Info box
    st.markdown("""
    <div class="info-box">
    <h4> About the Model</h4>
    <p><b>YOLOv8</b> is one of the best models for real-time object detection. 
    This program uses the YOLOv8-nano model by default.</p>
    </div>
    """, unsafe_allow_html=True)

# Main section
st.markdown('<h3 class="sub-header">Camera Mode</h3>', unsafe_allow_html=True)

# Model mapping
model_map = {
    "YOLOv8n (Small)": "yolov8n.pt",
    "YOLOv8s (Medium)": "yolov8s.pt",
    "YOLOv8m (Large)": "yolov8m.pt"
}

# Selected model
selected_model = model_map[model_type]

# Object detector
@st.cache_resource
def load_detector():
    return ObjectDetector()

detector = load_detector()

# Video stream configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Streamlit-webrtc component
class VideoProcessor:
    def __init__(self):
        self.detector = detector
        self.confidence = confidence_threshold
        self.iou = iou_threshold
        self.img_size = img_size
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Object detection
        results = self.detector.detect(
            img, 
            confidence=self.confidence,
            iou=self.iou,
            img_size=self.img_size
        )
        
        # Draw results
        processed_img = self.detector.draw_detections(img, results)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# Camera video stream
try:
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if not webrtc_ctx.state.playing:
        st.info("Click 'START' button to activate camera ")
        
except Exception as e:
    st.error(f"Camera error: {str(e)}")
    st.info("Camera input not available or permission not granted.")

# File upload alternative
st.markdown('<h3 class="sub-header"> Upload File</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'image':
        # Process image
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # Detection
        with st.spinner('Detecting objects...'):
            results = detector.detect(
                img_array,
                confidence=confidence_threshold,
                iou=iou_threshold,
                img_size=img_size
            )
        
        # Show results
        processed_img = detector.draw_detections(img_array, results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed_img, caption="Detected Objects", use_column_width=True)
        
        # Statistics
        if len(results) > 0:
            st.markdown(f"** Detected objects: {len(results)}**")
            for i, det in enumerate(results):
                st.write(f"{i+1}. {det['class']} - {det['confidence']:.2%} confidence")
    
    elif file_type == 'video':
        # Process video
        st.warning("Video processing feature is under development...")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Real-Time Object Detection Program | Using YOLOv8 model</p>
    <p>Built with Streamlit and OpenCV</p>
</div>
""", unsafe_allow_html=True)