import os # Added import
import torch
import numpy as np
import cv2
import logging
import time
import torchreid
from torchvision import transforms
from PIL import Image

class FeatureExtractor:
    """Person re-identification feature extractor using OSNet"""
    
    def __init__(self, config, logger=None):
        """Initialize the feature extractor
        
        Args:
            config (dict): Configuration dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger("person_reid")
        
        # Extract configuration
        self.model_path = config['reid']['model_path']
        self.device = config['system']['device']
        self.feature_dim = config['reid']['feature_dim']
        self.gallery_save_path = config['output'].get('gallery_save_path', None)
        if self.gallery_save_path and not os.path.exists(self.gallery_save_path):
            os.makedirs(self.gallery_save_path, exist_ok=True)
            self.logger.info(f"Created gallery save directory: {self.gallery_save_path}")
        
        # Initialize model
        self._initialize_model()
        
        # Set up image transformation
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _initialize_model(self):
        """Initialize the OSNet model"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model file not found: {self.model_path}. Will download from torchreid.")
            
            # Load model
            self.model = torchreid.utils.FeatureExtractor(
                model_name='osnet_x1_0',
                model_path=self.model_path,
                device=self.device
            )
            
            self.logger.info(f"Loaded OSNet model from {self.model_path} on {self.device}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize OSNet model: {str(e)}")
            raise
    
    def _extract_person_crops(self, frame, tracks):
        """Extract person crops from frame based on tracking bounding boxes
        
        Args:
            frame (numpy.ndarray): Input frame
            tracks (list): List of track dictionaries
        
        Returns:
            list: List of cropped person images
        """
        crops = []
        
        for track in tracks:
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Skip invalid bounding boxes
            if x1 >= x2 or y1 >= y2:
                self.logger.warning(f"Invalid bbox for track: {track.get('track_id', 'unknown')} bbox: {bbox} in frame of shape {frame.shape if hasattr(frame, 'shape') else 'unknown'}")
                crops.append(None)
                continue
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]

            # --- CLAHE Lighting Normalization ---
            # Convert to LAB color space
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            # Merge the channels back
            limg = cv2.merge((cl, a, b))
            # Convert back to BGR
            crop_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            # --- END CLAHE ---

            # Convert to PIL Image
            crop_pil = cv2.cvtColor(crop_clahe, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_pil)
            
            crops.append({'pil_image': crop_pil, 'cv_image': crop_clahe}) # Store both for flexibility
        
        return crops
    
    def _preprocess_crops(self, crops):
        """Preprocess person crops for feature extraction
        
        Args:
            crops (list): List of person crop dictionaries (containing PIL Images)
        
        Returns:
            torch.Tensor: Batch of preprocessed images
        """
        batch = []
        valid_indices = []
        
        for i, crop_dict in enumerate(crops):
            if crop_dict is None or crop_dict['pil_image'] is None:
                continue
                
            # Apply transformations
            try:
                img_tensor = self.transform(crop_dict['pil_image'])
                batch.append(img_tensor)
                valid_indices.append(i)
            except Exception as e:
                self.logger.warning(f"Failed to preprocess crop: {str(e)}")
        
        if not batch:
            return None, []
            
        # Stack into batch
        batch = torch.stack(batch)
        
        return batch, valid_indices
    
    def extract_features(self, frame, tracks):
        """Extract features from person tracks in a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            tracks (list): List of track dictionaries
        
        Returns:
            dict: Dictionary mapping track indices to feature vectors
        """
        if not tracks:
            return {}
            
        try:
            # Extract crops
            person_crops_data = self._extract_person_crops(frame, tracks)
            
            # Preprocess crops
            batch, valid_indices = self._preprocess_crops(person_crops_data)
            
            if batch is None:
                return {}, [] # Return empty list for crops if batch is None
            
            # Extract features
            start_time = time.time()
            with torch.no_grad():
                features = self.model(batch)
            process_time = time.time() - start_time
            
            # Convert to numpy and normalize
            features = features.cpu().numpy()
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            
            # Create result dictionary
            result = {}
            for i, idx in enumerate(valid_indices):
                result[idx] = features[i]
            
            self.logger.debug(f"Extracted features for {len(result)} persons in {process_time:.3f}s")
            
            # Return both features and the original crops data for saving
            return result, [person_crops_data[i] for i in valid_indices]
            
        except Exception as e:
            self.logger.error(f"Error in feature extraction: {str(e)}")
            return {}, [] # Return empty list for crops on error

    def save_person_crop(self, crop_cv_image, global_id, cam_id, frame_id, track_id):
        """Save a person crop to the gallery.

        Args:
            crop_cv_image (numpy.ndarray): The person crop image (OpenCV format).
            global_id (int or str): The global ID of the person.
            cam_id (int or str): The camera ID.
            frame_id (int): The frame ID.
            track_id (int): The track ID within the camera.
        """
        if not self.gallery_save_path:
            # self.logger.debug("Gallery save path not configured. Skipping crop saving.")
            return

        if crop_cv_image is None:
            self.logger.warning(f"Attempted to save None crop for GID {global_id}. Skipping.")
            return

        try:
            gid_folder = os.path.join(self.gallery_save_path, f"GID_{global_id}")
            os.makedirs(gid_folder, exist_ok=True)

            timestamp = int(time.time() * 1000) # Milliseconds for uniqueness
            filename = f"cam{cam_id}_frame{frame_id}_track{track_id}_{timestamp}.jpg"
            filepath = os.path.join(gid_folder, filename)
            
            cv2.imwrite(filepath, crop_cv_image)
            # self.logger.debug(f"Saved crop to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save crop for GID {global_id} to {gid_folder}: {str(e)}")