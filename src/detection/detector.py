import os
import torch
import numpy as np
from ultralytics import YOLO
import logging
import time

class PersonDetector:
    """Person detection module using YOLOv8"""
    
    def __init__(self, config, logger=None):
        """Initialize the person detector
        
        Args:
            config (dict): Configuration dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger("person_reid")
        
        # Extract configuration
        self.model_path = config['detection']['model_path']
        self.conf_threshold = config['detection']['confidence_threshold']
        self.device = config['system']['device']
        self.image_size = config['detection']['image_size']
        self.half_precision = config['detection']['half_precision']
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the YOLOv8 model"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model file not found: {self.model_path}. Will download from Ultralytics.")
            
            # Load model
            self.model = YOLO(self.model_path)
            
            # Set device
            self.model.to(self.device)
            
            # Set half precision if enabled and supported
            if self.half_precision and self.device != 'cpu' and torch.cuda.is_available():
                self.model.model.half()
                self.logger.info("Using half precision for detection")
            
            self.logger.info(f"Loaded YOLOv8 model from {self.model_path} on {self.device}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLOv8 model: {str(e)}")
            raise
    
    def detect(self, frames):
        """Detect persons in frames
        
        Args:
            frames (dict): Dictionary of frame data indexed by camera ID
        
        Returns:
            dict: Dictionary of detection results indexed by camera ID
        """
        results = {}
        
        # Process each camera's frame
        for cam_id, frame_data in frames.items():
            frame = frame_data['frame']
            frame_id = frame_data['frame_id']
            
            try:
                # Run detection
                start_time = time.time()
                detections = self.model(
                    frame, 
                    conf=self.conf_threshold,
                    imgsz=self.image_size,
                    classes=0,  # Only detect persons (class 0 in COCO)
                    verbose=False
                )
                process_time = time.time() - start_time
                
                # Extract detection results
                boxes = []
                for detection in detections:
                    # Convert to numpy
                    boxes_data = detection.boxes.data.cpu().numpy()
                    
                    # Filter for person class (class 0)
                    person_boxes = boxes_data[boxes_data[:, 5] == 0]
                    
                    for box in person_boxes:
                        x1, y1, x2, y2, conf, _ = box
                        boxes.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'frame_id': frame_id,
                            'cam_id': cam_id
                        })
                
                results[cam_id] = {
                    'detections': boxes,
                    'frame_id': frame_id,
                    'cam_id': cam_id,
                    'process_time': process_time
                }
                
                self.logger.debug(f"Detected {len(boxes)} persons in frame {frame_id} "
                                 f"from camera {cam_id} in {process_time:.3f}s")
            
            except Exception as e:
                self.logger.error(f"Error in detection for camera {cam_id}: {str(e)}")
                results[cam_id] = {
                    'detections': [],
                    'frame_id': frame_id,
                    'cam_id': cam_id,
                    'error': str(e)
                }
        
        return results 