# Real-Time Object Detection Web App

A powerful and user-friendly Streamlit-based web application for real-time object detection using the YOLOv8 algorithm. This project supports both live camera feeds and static image analysis with customizable parameters.

##  Features
- Real-time Detection: High-speed object detection via your webcam.
-  Image Analysis: Upload images (JPG/PNG) to detect and analyze objects.
-  Multiple Model Support: Choose between YOLOv8 nano, small, and medium models.
-  Dynamic Configuration: Adjust Confidence threshold, NMS, and Image size on the fly.
-  Statistical Insights: View real-time results and detection statistics.
-  Docker Support: Fully containerized for easy deployment.

##  Installation: 

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ceyhunmmmdov204-lgtm/real-time-object-detection.git
   cd real-time-object-detection

Install required libraries:

Bash

pip install -r requirements.txt
Download pre-trained models:

Bash

python models/download_models.py
Run the application:

Bash

streamlit run app.py
Installation with Docker
Build the Docker image:

Bash

docker build -t object-detection-app .
Run the container:

Bash

docker run -p 8501:8501 --device=/dev/video0:/dev/video0 object-detection-app
Note: Use docker-compose up -d as an alternative if using Compose.

 Usage
Open your browser and navigate to http://localhost:8501.

Click the "START" button to activate your camera.

Use the left sidebar to configure model parameters and switch between models.

Use the file upload section for individual image analysis.


 Project Structure:
Plaintext

real-time-object-detection/

├── app.py                       # Main Streamlit application

├── object_detector.py           # Object detection logic

├── requirements.txt          # Python libraries

├── Dockerfile               # Docker configuration

├── docker-compose.yml       # Docker Compose file

├── download_models.py       # Model weights and download script

└── visualization.py         # Helper functions for visualization
 
 Configuration:
Model Type: YOLOv8-nano (fastest), small, or medium (most accurate).

Confidence Threshold: 0.0 to 1.0 (Default: 0.5).

NMS Threshold: 0.0 to 1.0 (Default: 0.45).

Image Size: 320 to 1280 pixels (Default: 640).

 
 Contributing:
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.


License:

This project is licensed under the MIT License.
