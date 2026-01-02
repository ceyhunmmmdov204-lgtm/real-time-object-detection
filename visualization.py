import cv2
import numpy as np
from typing import List, Dict, Any

def create_color_palette(n_colors: int) -> List[tuple]:
    """Create color palette"""
    colors = []
    for i in range(n_colors):
        hue = i * 255 // n_colors
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors

def draw_statistics(image: np.ndarray, 
                    detections: List[Dict[str, Any]], 
                    fps: float = None) -> np.ndarray:
    """
    Display statistics on image
    """
    output_image = image.copy()
    height, width = image.shape[:2]
    
    # Statistics panel
    stats_bg = np.zeros((100, width, 3), dtype=np.uint8)
    stats_bg.fill(50)  # Dark gray background
    
    # Texts
    text_y = 30
    text_color = (255, 255, 255)
    
    # Object count
    object_count = len(detections)
    count_text = f"Objects: {object_count}"
    cv2.putText(stats_bg, count_text, (20, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # FPS (if available)
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(stats_bg, fps_text, (width - 150, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Object types
    if object_count > 0:
        classes = set([det['class'] for det in detections])
        classes_text = f"Object types: {', '.join(classes)}"
        cv2.putText(stats_bg, classes_text[:100], (20, text_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Combine with panel
    output_image = np.vstack([output_image, stats_bg])
    
    return output_image