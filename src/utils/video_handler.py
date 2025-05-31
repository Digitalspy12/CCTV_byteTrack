import os
import cv2
import numpy as np
import logging
import threading
from collections import deque

class VideoHandler:
    """Video input handler for processing multiple video streams"""
    
    def __init__(self, video_paths, config, logger=None):
        """Initialize the video handler
        
        Args:
            video_paths (list): List of paths to video files
            config (dict): Configuration dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.video_paths = video_paths
        self.config = config
        self.logger = logger or logging.getLogger("person_reid")
        
        self.captures = {}
        self.frame_buffers = {}
        self.frame_counts = {}
        self.fps = {}
        self.dimensions = {}
        self.is_running = True
        
        self.frame_skip = config['video']['frame_skip']
        self.resize_factor = config['video']['resize_factor']
        self.max_buffer_size = config['video']['max_frame_buffer']
        
        # Initialize video captures
        self._initialize_videos()
        
        # Start reading threads
        self.threads = []
        for cam_id in range(len(self.video_paths)):
            thread = threading.Thread(
                target=self._read_frames_thread, 
                args=(cam_id,),
                daemon=True
            )
            self.threads.append(thread)
            thread.start()
    
    def _initialize_videos(self):
        """Initialize video capture objects for all video paths"""
        for cam_id, video_path in enumerate(self.video_paths):
            if not os.path.exists(video_path):
                self.logger.error(f"Video file not found: {video_path}")
                continue
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}")
                continue
                
            # Store capture object
            self.captures[cam_id] = cap
            
            # Get video properties
            self.frame_counts[cam_id] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps[cam_id] = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.dimensions[cam_id] = (width, height)
            
            # Initialize frame buffer
            self.frame_buffers[cam_id] = deque(maxlen=self.max_buffer_size)
            
            self.logger.info(f"Initialized video {cam_id}: {video_path}, "
                            f"Frames: {self.frame_counts[cam_id]}, "
                            f"FPS: {self.fps[cam_id]}, "
                            f"Resolution: {width}x{height}")
    
    def _read_frames_thread(self, cam_id):
        """Thread function to continuously read frames from a video
        
        Args:
            cam_id (int): Camera ID
        """
        if cam_id not in self.captures:
            return
            
        cap = self.captures[cam_id]
        frame_count = 0
        
        while self.is_running:
            # Skip frames if needed
            for _ in range(self.frame_skip):
                success = cap.grab()
                if not success:
                    break
                frame_count += 1
            
            # Read the frame
            success, frame = cap.read()
            if not success:
                # End of video
                if frame_count >= self.frame_counts[cam_id] - 1:
                    self.logger.info(f"Reached end of video {cam_id}")
                else:
                    self.logger.warning(f"Failed to read frame from video {cam_id}")
                break
            
            frame_count += 1
            
            # Resize if needed
            if self.resize_factor != 1.0:
                width = int(frame.shape[1] * self.resize_factor)
                height = int(frame.shape[0] * self.resize_factor)
                frame = cv2.resize(frame, (width, height))
            
            # Add to buffer
            self.frame_buffers[cam_id].append({
                'frame': frame,
                'frame_id': frame_count,
                'cam_id': cam_id,
                'timestamp': frame_count / self.fps[cam_id]
            })
    
    def get_frames(self):
        """Get synchronized frames from all cameras
        
        Returns:
            dict: Dictionary of frame data indexed by camera ID
        """
        frames = {}
        
        # Check if any buffers are empty
        empty_buffers = [cam_id for cam_id, buffer in self.frame_buffers.items() if not buffer]
        if empty_buffers:
            # Wait for buffers to fill
            return None
        
        # Get one frame from each camera
        for cam_id in self.frame_buffers:
            if self.frame_buffers[cam_id]:
                frames[cam_id] = self.frame_buffers[cam_id].popleft()
        
        return frames
    
    def release(self):
        """Release all video captures and stop threads"""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Release captures
        for cap in self.captures.values():
            cap.release()
        
        self.logger.info("Released all video captures") 