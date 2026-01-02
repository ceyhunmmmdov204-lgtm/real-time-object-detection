import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import time
from typing import List, Dict, Any

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Object detector using YOLOv8 model
        """
        self.model_name = model_name
        self.model = None
        self.class_names = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
    
    def load_model(self):
        """Load the model"""
        try:
            # Download model if not exists
            if not os.path.exists(self.model_name):
                print(f"Model {self.model_name} not found, downloading...")
                self.model = YOLO(self.model_name)
            else:
                self.model = YOLO(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.class_names = self.model.names
            print(f"Model {self.model_name} loaded successfully")
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Model loading error: {e}")
            raise
    
    def detect(self, 
               image: np.ndarray, 
               confidence: float = 0.5,
               iou: float = 0.45,
               img_size: int = 640) -> List[Dict[str, Any]]:
        """
        Detect objects in image
        
        Args:
            image: Image array (BGR format)
            confidence: Confidence threshold
            iou: NMS threshold
            img_size: Processing size
        
        Returns:
            List of detected objects
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Detection with YOLOv8
            results = self.model(
                image, 
                conf=confidence,
                iou=iou,
                imgsz=img_size,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class': class_name,
                            'class_id': class_id
                        })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detected objects on image
        
        Args:
            image: Original image
            detections: List of detected objects
        
        Returns:
            Image with drawn detections
        """
        if len(detections) == 0:
            return image
        
        output_image = image.copy()
        
        # Color palette
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
        ]
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class']
            confidence = det['confidence']
            class_id = det['class_id']
            
            # Select color
            color = colors[class_id % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Text background
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Text background rectangle
            cv2.rectangle(
                output_image, 
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                output_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return output_image
    
    def process_frame(self, 
                     frame: np.ndarray,
                     confidence: float = 0.5,
                     iou: float = 0.45,
                     img_size: int = 640) -> tuple:
        """
        Process a video frame
        
        Returns:
            (processed_frame, detections)
        """
        detections = self.detect(frame, confidence, iou, img_size)
        processed_frame = self.draw_detections(frame, detections)
        
        return processed_frame, detections