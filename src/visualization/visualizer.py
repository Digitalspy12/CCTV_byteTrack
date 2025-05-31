import cv2
import numpy as np
import logging
import os
import time
import colorsys
import platform
import sys

class Visualizer:
    """Visualization module for person re-identification results"""
    
    def __init__(self, config, logger=None):
        """Initialize the visualizer
        
        Args:
            config (dict): Configuration dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger("person_reid")
        
        # Extract configuration
        self.output_dir = config['output'].get('output_dir', 'data/output')
        self.save_video = config['output'].get('video_output', True)
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize video writers
        self.video_writers = {}
        
        # Initialize color map for global IDs
        self.color_map = {}
        
        # Check headless mode configuration
        headless_mode = config['output'].get('headless_mode', False)
        
        # Detect headless environment
        self.is_headless = headless_mode or self._detect_headless()
        if self.is_headless:
            self.logger.warning("Running in headless environment. GUI visualization disabled.")
        
        self.fps_map = {}  # cam_id -> fps
    
    def _detect_headless(self):
        """Detect if running in a headless environment
        
        Returns:
            bool: True if running in a headless environment
        """
        # Check for common headless environments
        if 'google.colab' in sys.modules:
            return True
            
        # Check for DISPLAY environment variable on Unix-like systems
        if platform.system() != "Windows" and not os.environ.get('DISPLAY'):
            return True
            
        # Try to create a test window
        try:
            test_window_name = "__test_window__"
            cv2.namedWindow(test_window_name, cv2.WINDOW_AUTOSIZE)
            cv2.destroyWindow(test_window_name)
            return False
        except:
            return True
    
    def _get_color(self, global_id):
        """Get color for a global ID
        
        Args:
            global_id (int): Global person ID
        
        Returns:
            tuple: BGR color
        """
        if global_id not in self.color_map:
            # Generate distinct color using HSV
            hue = (global_id * 0.1) % 1.0
            sat = 0.8
            val = 0.9
            
            # Convert to RGB and then to BGR
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            self.color_map[global_id] = (int(b * 255), int(g * 255), int(r * 255))
        
        return self.color_map[global_id]
    
    def draw_tracks(self, frame, tracks):
        """Draw tracks on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            tracks (list): List of track dictionaries
        
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            global_id = track.get('global_id')
            
            # Draw bounding box
            color = self._get_color(global_id) if global_id is not None else (0, 255, 0)
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw IDs
            label = f"P{global_id}" if global_id is not None else f"T{track_id}"
            cv2.putText(
                vis_frame, 
                label, 
                (bbox[0], bbox[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
        
        return vis_frame
    
    def add_frame_info(self, frame, frame_data, stats=None):
        """Add frame information to visualization
        
        Args:
            frame (numpy.ndarray): Input frame
            frame_data (dict): Frame data dictionary
            stats (dict, optional): Statistics dictionary
        
        Returns:
            numpy.ndarray: Frame with information
        """
        cam_id = frame_data['cam_id']
        frame_id = frame_data['frame_id']
        
        # Add camera and frame info
        cv2.putText(
            frame,
            f"Camera: {cam_id}, Frame: {frame_id}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )
        
        # Add stats if available
        if stats:
            y_pos = 40
            for key, value in stats.items():
                if key == 'persons_by_camera':
                    continue
                    
                cv2.putText(
                    frame,
                    f"{key}: {value}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
                y_pos += 20
        
        return frame
    
    def display(self, frame, cam_id, window_name=None):
        """Display frame in a window
        
        Args:
            frame (numpy.ndarray): Frame to display
            cam_id (int): Camera ID
            window_name (str, optional): Window name
        """
        # Skip display in headless environment
        if self.is_headless:
            return
            
        if window_name is None:
            window_name = f"Camera {cam_id}"
            
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    
    def save_frame(self, frame, frame_data):
        """Save frame to output directory
        
        Args:
            frame (numpy.ndarray): Frame to save
            frame_data (dict): Frame data dictionary
        """
        cam_id = frame_data['cam_id']
        frame_id = frame_data['frame_id']
        
        # Create output directory for camera
        cam_dir = os.path.join(self.output_dir, f"camera_{cam_id}")
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir, exist_ok=True)
        
        # Save frame
        frame_path = os.path.join(cam_dir, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(frame_path, frame)
    
    def set_fps(self, cam_id, fps):
        """Set FPS for a specific camera"""
        self.fps_map[cam_id] = fps

    def init_video_writer(self, frame_shape, cam_id, fps=None):
        """Initialize video writer for a camera
        
        Args:
            frame_shape (tuple): Frame shape (height, width, channels)
            cam_id (int): Camera ID
            fps (int, optional): Frames per second
        """
        if not self.save_video:
            return
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.output_dir, f"camera_{cam_id}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frame_shape[:2]
        # Use provided fps, or from map, or default to 30
        input_fps = fps if fps is not None else self.fps_map.get(cam_id, 30)
        frame_skip = self.config['video'].get('frame_skip', 0)
        if frame_skip > 0:
            fps_to_use = input_fps / (frame_skip + 1)
        else:
            fps_to_use = input_fps
        self.video_writers[cam_id] = cv2.VideoWriter(
            video_path, 
            fourcc, 
            fps_to_use, 
            (width, height)
        )
        self.logger.info(f"Initialized video writer for camera {cam_id}: {video_path} (FPS: {fps_to_use})")
    
    def write_frame(self, frame, cam_id):
        """Write frame to video
        
        Args:
            frame (numpy.ndarray): Frame to write
            cam_id (int): Camera ID
        """
        if not self.save_video:
            return
        if cam_id not in self.video_writers:
            self.init_video_writer(frame.shape, cam_id)
        self.video_writers[cam_id].write(frame)
    
    def release(self):
        """Release all video writers"""
        for writer in self.video_writers.values():
            writer.release()
        
        # Only destroy windows if not in headless mode
        if not self.is_headless:
            cv2.destroyAllWindows()
        
        self.logger.info("Released all video writers") 