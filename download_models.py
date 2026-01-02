import os
from ultralytics import YOLO

def download_models():
    """Download YOLOv8 models"""
    models = {
        'nano': 'yolov8n.pt',
        'small': 'yolov8s.pt',
        'medium': 'yolov8m.pt',
        'large': 'yolov8l.pt',
        'xlarge': 'yolov8x.pt'
    }
    
    print("Downloading YOLOv8 models...")
    
    for size, model_name in models.items():
        if not os.path.exists(model_name):
            print(f"Downloading: {model_name}")
            try:
                model = YOLO(model_name)
                print(f"✓ {model_name} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
        else:
            print(f"✓ {model_name} already exists")
    
    print("\nModel download completed!")

if __name__ == "__main__":
    download_models()