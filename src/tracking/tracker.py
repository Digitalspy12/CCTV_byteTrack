import numpy as np
import logging
from collections import defaultdict
import time
import scipy.optimize as optimize
from filterpy.kalman import KalmanFilter # Added import

class Track:
    """Track class for a single object"""
    
    _next_id = 1  # Class variable for track ID assignment
    
    def __init__(self, bbox, confidence, frame_id, cam_id):
        """Initialize a new track
        
        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            confidence (float): Detection confidence
            frame_id (int): Frame ID
            cam_id (int): Camera ID
        """
        self.track_id = Track._next_id
        Track._next_id += 1
        
        self.bbox = bbox
        self.confidence = confidence
        self.frame_id = frame_id
        self.cam_id = cam_id
        
        # Track state
        self.state = "new"  # new, tracked, lost, removed
        self.age = 0
        self.time_since_update = 0
        self.max_age = 30  # Maximum frames to keep a lost track
        
        # Track history
        self.bboxes = [bbox]
        self.confidences = [confidence]
        self.frame_ids = [frame_id]
        
        # For re-identification
        self.features = []
        self.global_id = None

        # Initialize Kalman Filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # State vector [x, y, vx, vy] (center_x, center_y, velocity_x, velocity_y)
        # Measurement vector [x, y] (center_x, center_y)
        
        # State transition matrix (assuming constant velocity)
        self.kf.F = np.array([[1, 0, 1, 0],  # x = x + vx*dt (dt=1 frame)
                              [0, 1, 0, 1],  # y = y + vy*dt
                              [0, 0, 1, 0],  # vx = vx
                              [0, 0, 0, 1]]) # vy = vy
        
        # Measurement function (maps state to measurement)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        
        # Measurement noise covariance matrix (R)
        # Adjust based on detection noise
        self.kf.R *= np.array([[10**2, 0],
                               [0, 10**2]]) 
        
        # Process noise covariance matrix (Q)
        # Accounts for uncertainty in the motion model
        self.kf.Q = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 10, 0],
                              [0, 0, 0, 10]]) * 0.1

        # Initial state covariance matrix (P)
        self.kf.P *= 1000.  # High initial uncertainty
        
        # Initial state (center_x, center_y, 0, 0)
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.kf.x = np.array([center_x, center_y, 0, 0]).reshape(4, 1)
    
    def update(self, bbox, confidence, frame_id):
        """Update track with new detection
        
        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            confidence (float): Detection confidence
            frame_id (int): Frame ID
        """
        self.bbox = bbox
        self.confidence = confidence
        self.frame_id = frame_id
        
        self.state = "tracked"
        self.time_since_update = 0
        self.age += 1
        
        # Update history
        self.bboxes.append(bbox)
        self.confidences.append(confidence)
        self.frame_ids.append(frame_id)
        
        # Limit history length
        if len(self.bboxes) > 30:
            self.bboxes.pop(0)
            self.confidences.pop(0)
            self.frame_ids.pop(0)

        # Update Kalman Filter
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        measurement = np.array([center_x, center_y]).reshape(2, 1)
        self.kf.update(measurement)

        # Update bbox from KF state
        kf_center_x, kf_center_y = self.kf.x[0, 0], self.kf.x[1, 0]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        self.bbox = [
            int(kf_center_x - width / 2),
            int(kf_center_y - height / 2),
            int(kf_center_x + width / 2),
            int(kf_center_y + height / 2)
        ]
    
    def predict(self):
        """Predict next position using Kalman Filter"""
        self.kf.predict()
        
        # Get predicted center coordinates
        center_x, center_y = self.kf.x[0, 0], self.kf.x[1, 0]
        
        # Assume width and height remain similar to the last known bbox
        if self.bboxes:
            last_bbox = self.bboxes[-1]
            width = last_bbox[2] - last_bbox[0]
            height = last_bbox[3] - last_bbox[1]
        else: # Should not happen if initialized correctly
            width, height = 10, 20 # Default small size
            
        # Construct predicted bounding box
        pred_box = [
            int(center_x - width / 2),
            int(center_y - height / 2),
            int(center_x + width / 2),
            int(center_y + height / 2)
        ]
        return pred_box
    
    def mark_lost(self):
        """Mark track as lost"""
        self.state = "lost"
        self.time_since_update += 1
    
    def mark_removed(self):
        """Mark track as removed"""
        self.state = "removed"
    
    def is_confirmed(self):
        """Check if track is confirmed (tracked for enough frames)"""
        return self.age >= 3 and self.state == "tracked"
    
    def is_lost(self):
        """Check if track is lost"""
        return self.state == "lost"
    
    def is_removed(self):
        """Check if track is removed"""
        return self.state == "removed"
    
    def should_remove(self):
        """Check if track should be removed"""
        return self.time_since_update > self.max_age


class MultiObjectTracker:
    """Multi-object tracker based on ByteTrack principles"""
    
    def __init__(self, config, logger=None):
        """Initialize the tracker
        
        Args:
            config (dict): Configuration dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger("person_reid")
        
        # Extract configuration
        self.track_thresh = config['tracking']['track_thresh']
        self.track_buffer = config['tracking']['track_buffer']
        self.match_thresh = config['tracking']['match_thresh']
        
        # Initialize trackers for each camera
        self.trackers = {}
        self.tracks_by_camera = defaultdict(list)
    
    def _reset_track_ids(self):
        """Reset the track ID counter"""
        Track._next_id = 1
    
    def _get_tracker(self, cam_id):
        """Get tracker for specific camera
        
        Args:
            cam_id (int): Camera ID
        
        Returns:
            dict: Tracker state for the camera
        """
        if cam_id not in self.trackers:
            self.trackers[cam_id] = {
                'tracks': [],
                'lost_tracks': [],
                'removed_tracks': []
            }
        
        return self.trackers[cam_id]
    
    def _iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes
        
        Args:
            bbox1 (list): First bounding box [x1, y1, x2, y2]
            bbox2 (list): Second bounding box [x1, y1, x2, y2]
        
        Returns:
            float: IoU score
        """
        # Get the coordinates of bounding boxes
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Get the coordinates of intersection
        xx1 = max(x1_min, x2_min)
        yy1 = max(y1_min, y2_min)
        xx2 = min(x1_max, x2_max)
        yy2 = min(y1_max, y2_max)
        
        # Check if there is an intersection
        if xx2 < xx1 or yy2 < yy1:
            return 0.0
        
        # Area of intersection
        intersection = (xx2 - xx1) * (yy2 - yy1)
        
        # Area of both bounding boxes
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        # IoU = intersection / union
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_cost_matrix(self, tracks, detections):
        """Compute cost matrix between tracks and detections
        
        Args:
            tracks (list): List of Track objects
            detections (list): List of detection dictionaries
        
        Returns:
            numpy.ndarray: Cost matrix
        """
        num_tracks = len(tracks)
        num_detections = len(detections)
        
        if num_tracks == 0 or num_detections == 0:
            return np.empty((0, 0))
        
        cost_matrix = np.zeros((num_tracks, num_detections))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Calculate IoU cost (1 - IoU)
                iou = self._iou(track.bbox, det['bbox'])
                cost_matrix[i, j] = 1 - iou
        
        return cost_matrix
    
    def _associate_detections_to_tracks(self, tracks, detections, threshold):
        """Associate detections to tracks using Hungarian algorithm
        
        Args:
            tracks (list): List of Track objects
            detections (list): List of detection dictionaries
            threshold (float): Matching threshold
        
        Returns:
            tuple: Matched indices, unmatched track indices, unmatched detection indices
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(tracks, detections)
        
        # Apply Hungarian algorithm
        row_indices, col_indices = optimize.linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= threshold:
                matches.append((row, col))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def update(self, detection_results):
        """Update tracks with new detections
        
        Args:
            detection_results (dict): Detection results indexed by camera ID
        
        Returns:
            dict: Updated tracks indexed by camera ID
        """
        results = {}
        
        # Process each camera
        for cam_id, result in detection_results.items():
            frame_id = result['frame_id']
            detections = result['detections']
            
            # Get tracker for this camera
            tracker = self._get_tracker(cam_id)
            tracks = tracker['tracks']
            lost_tracks = tracker['lost_tracks']
            
            # Predict new locations of tracks
            for track in tracks + lost_tracks:
                track.predict()
            
            # Split detections by confidence
            high_conf_dets = [d for d in detections if d['confidence'] >= self.track_thresh]
            low_conf_dets = [d for d in detections if d['confidence'] < self.track_thresh]
            
            # First association with high confidence detections
            matches_a, unmatched_tracks_a, unmatched_dets_a = self._associate_detections_to_tracks(
                tracks, high_conf_dets, 1 - self.match_thresh
            )
            
            # Update matched tracks
            for track_idx, det_idx in matches_a:
                track = tracks[track_idx]
                det = high_conf_dets[det_idx]
                track.update(det['bbox'], det['confidence'], frame_id)
            
            # Try to recover lost tracks
            if len(unmatched_dets_a) > 0 and len(lost_tracks) > 0:
                matches_b, unmatched_lost_a, unmatched_dets_b = self._associate_detections_to_tracks(
                    lost_tracks, [high_conf_dets[i] for i in unmatched_dets_a], 1 - self.match_thresh
                )
                
                # Recover matched tracks
                for track_idx, det_idx in matches_b:
                    track = lost_tracks[track_idx]
                    det = high_conf_dets[unmatched_dets_a[det_idx]]
                    track.update(det['bbox'], det['confidence'], frame_id)
                    track.state = "tracked"
                    tracks.append(track)
                
                # Remove recovered tracks from lost_tracks
                for track_idx, _ in sorted(matches_b, key=lambda x: x[0], reverse=True):
                    lost_tracks.pop(track_idx)
                
                unmatched_dets_a = [unmatched_dets_a[i] for i in unmatched_dets_b]
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets_a:
                det = high_conf_dets[det_idx]
                track = Track(det['bbox'], det['confidence'], frame_id, cam_id)
                tracks.append(track)
            
            # Mark unmatched tracks as lost
            for track_idx in unmatched_tracks_a:
                track = tracks[track_idx]
                track.mark_lost()
                lost_tracks.append(track)
            
            # Remove tracks from active list
            tracks = [t for t in tracks if not t.is_lost()]
            
            # Second association with low confidence detections
            if len(low_conf_dets) > 0:
                matches_c, _, _ = self._associate_detections_to_tracks(
                    lost_tracks, low_conf_dets, 1 - 0.5  # Lower threshold for low confidence
                )
                
                # Update matched tracks
                for track_idx, det_idx in matches_c:
                    track = lost_tracks[track_idx]
                    det = low_conf_dets[det_idx]
                    track.update(det['bbox'], det['confidence'], frame_id)
                    track.state = "tracked"
                    tracks.append(track)
                
                # Remove recovered tracks from lost_tracks
                for track_idx, _ in sorted(matches_c, key=lambda x: x[0], reverse=True):
                    lost_tracks.pop(track_idx)
            
            # Update lost tracks
            for track in lost_tracks:
                track.time_since_update += 1
            
            # Remove old lost tracks
            lost_tracks = [t for t in lost_tracks if not t.should_remove()]
            removed_tracks = [t for t in lost_tracks if t.should_remove()]
            
            # Update tracker state
            tracker['tracks'] = tracks
            tracker['lost_tracks'] = lost_tracks
            tracker['removed_tracks'].extend(removed_tracks)
            
            # Prepare results
            active_tracks = []
            for track in tracks:
                if track.is_confirmed():
                    active_tracks.append({
                        'track_id': track.track_id,
                        'global_id': track.global_id,
                        'bbox': track.bbox,
                        'confidence': track.confidence,
                        'state': track.state,
                        'age': track.age,
                        'time_since_update': track.time_since_update,
                        'cam_id': track.cam_id,
                        'frame_id': track.frame_id
                    })
            
            results[cam_id] = {
                'tracks': active_tracks,
                'frame_id': frame_id,
                'cam_id': cam_id,
                'num_tracks': len(active_tracks)
            }
            
            self.logger.debug(f"Camera {cam_id}, Frame {frame_id}: "
                             f"{len(active_tracks)} active tracks, "
                             f"{len(lost_tracks)} lost tracks")
        
        return results