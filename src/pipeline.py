import os
import time
import logging
import argparse
import cv2
import numpy as np
import json

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.utils.video_handler import VideoHandler
from src.detection.detector import PersonDetector
from src.tracking.tracker import MultiObjectTracker
from src.reid.feature_extractor import FeatureExtractor
from src.reid.reid_manager import ReIDManager
from src.visualization.visualizer import Visualizer

class PersonReIDPipeline:
    """Main pipeline for person re-identification system"""
    
    def __init__(self, config_path):
        """Initialize the pipeline
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Set up logger
        log_level = self.config['system']['log_level']
        self.logger = setup_logger(log_level)
        
        self.logger.info("Initializing Person Re-ID Pipeline")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components"""
        # Detection
        self.logger.info("Initializing person detector")
        self.detector = PersonDetector(self.config, self.logger)
        
        # Tracking
        self.logger.info("Initializing multi-object tracker")
        self.tracker = MultiObjectTracker(self.config, self.logger)
        
        # Feature extraction
        self.logger.info("Initializing feature extractor")
        self.feature_extractor = FeatureExtractor(self.config, self.logger)
        
        # Re-ID manager
        self.logger.info("Initializing re-identification manager")
        self.reid_manager = ReIDManager(self.config, self.logger)
        
        # Visualization
        self.logger.info("Initializing visualizer")
        self.visualizer = Visualizer(self.config, self.logger)
    
    def process_videos(self, video_paths):
        """Process multiple videos
        
        Args:
            video_paths (list): List of video file paths
        """
        # Initialize video handler
        self.logger.info(f"Processing {len(video_paths)} videos")
        video_handler = VideoHandler(video_paths, self.config, self.logger)
        
        # Set correct FPS for each camera in the visualizer
        for cam_id, fps in video_handler.fps.items():
            self.visualizer.set_fps(cam_id, fps)
        
        # Main processing loop
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Get synchronized frames
                frames = video_handler.get_frames()
                if frames is None:
                    # Wait for frames
                    time.sleep(0.01)
                    continue
                
                if not frames:
                    # End of videos
                    break
                
                frame_count += 1
                
                # Process frames
                self.logger.debug(f"Processing frame {frame_count}")
                
                # 1. Person detection
                detection_results = self.detector.detect(frames)
                
                # 2. Object tracking
                tracking_results = self.tracker.update(detection_results)
                
                # 3. Feature extraction and re-identification
                for cam_id, result in tracking_results.items():
                    frame_data = frames[cam_id]
                    tracks = result['tracks']
                    
                    if not tracks:
                        continue
                    
                    # Extract features
                    features_dict, extracted_crops_data = self.feature_extractor.extract_features(frame_data['frame'], tracks)
                    
                    # Add features to tracks
                    tracks_with_features = []
                    # Keep track of which crop corresponds to which track
                    track_to_crop_map = {}
                    for i, track in enumerate(tracks):
                        if i in features_dict:
                            track['feature'] = features_dict[i]
                            tracks_with_features.append(track)
                            # Assuming extracted_crops_data is ordered consistently with features_dict keys
                            # Find the original crop data for this track
                            # This relies on the assumption that valid_indices from feature_extractor maps correctly
                            # A more robust way might be to pass track_id into feature_extractor or ensure indices align
                            original_crop_index = -1
                            current_feature_dict_key_count = 0
                            for k_idx, key in enumerate(sorted(features_dict.keys())):
                                if key == i:
                                    original_crop_index = current_feature_dict_key_count
                                    break
                                current_feature_dict_key_count +=1
                            
                            if original_crop_index != -1 and original_crop_index < len(extracted_crops_data):
                                track_to_crop_map[track['track_id']] = extracted_crops_data[original_crop_index]['cv_image']
                            else:
                                track_to_crop_map[track['track_id']] = None # Should not happen if logic is correct

                    # Update re-identification
                    if tracks_with_features:
                        updated_tracks_with_gids = self.reid_manager.update(tracks_with_features, frame_data)
                        
                        # Update tracks with global IDs and save crops
                        for updated_track in updated_tracks_with_gids:
                            for original_track in tracks: # Find the original track to update its GID
                                if original_track['track_id'] == updated_track['track_id']:
                                    original_track['global_id'] = updated_track['global_id']
                                    # Save crop if GID is assigned and crop is available
                                    if 'global_id' in original_track and original_track['global_id'] is not None:
                                        crop_image_to_save = track_to_crop_map.get(original_track['track_id'])
                                        if crop_image_to_save is not None:
                                            self.feature_extractor.save_person_crop(
                                                crop_image_to_save,
                                                original_track['global_id'],
                                                original_track['cam_id'],
                                                original_track['frame_id'],
                                                original_track['track_id']
                                            )
                                        else:
                                            self.logger.warning(f"Crop not found for track {original_track['track_id']} for saving.")
                                    break # Found and updated original_track
                    
                    # Visualization
                    if self.config['output']['visualization']:
                        # Draw tracks
                        vis_frame = self.visualizer.draw_tracks(frame_data['frame'], tracks)
                        
                        # Add frame info
                        stats = self.reid_manager.get_stats()
                        vis_frame = self.visualizer.add_frame_info(vis_frame, frame_data, stats)
                        
                        # Display
                        self.visualizer.display(vis_frame, cam_id)
                        
                        # Save video
                        if self.config['output']['video_output']:
                            self.visualizer.write_frame(vis_frame, cam_id)
                
                # Process at most 30 FPS
                elapsed = time.time() - start_time
                if elapsed < frame_count / 30:
                    time.sleep(frame_count / 30 - elapsed)
                
                # Check for exit key - only if not in headless mode
                if not self.visualizer.is_headless and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Error in processing: {str(e)}", exc_info=True)
        
        finally:
            # Clean up
            video_handler.release()
            self.visualizer.release()
            
            # Print statistics
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            self.logger.info(f"Processed {frame_count} frames in {elapsed:.2f}s ({fps:.2f} FPS)")
            
            # Print re-ID statistics
            stats = self.reid_manager.get_stats()
            self.logger.info(f"Re-ID statistics: {stats}")
    
    def save_results(self, output_path):
        """Save processing results to file
        
        Args:
            output_path (str): Path to output file
        """
        if not self.config['output']['save_results']:
            return
            
        # Get statistics
        stats = self.reid_manager.get_stats()
        
        # Create result dictionary
        results = {
            'stats': stats,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': self.config
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Saved results to {output_path}")

    def stream_processed_frames(self, video_paths):
        """
        Generator that yields processed frames with overlays for each camera.
        Suitable for real-time streaming in Streamlit.
        Args:
            video_paths (list): List of video file paths
        Yields:
            dict: {cam_id: vis_frame} for each synchronized set of frames
        """
        video_handler = VideoHandler(video_paths, self.config, self.logger)
        for cam_id, fps in video_handler.fps.items():
            self.visualizer.set_fps(cam_id, fps)
        frame_count = 0
        start_time = time.time()
        try:
            while True:
                frames = video_handler.get_frames()
                if frames is None:
                    time.sleep(0.01)
                    continue
                if not frames:
                    break
                frame_count += 1
                detection_results = self.detector.detect(frames)
                tracking_results = self.tracker.update(detection_results)
                vis_frames = {}
                for cam_id, result in tracking_results.items():
                    frame_data = frames[cam_id]
                    tracks = result['tracks']
                    if not tracks:
                        vis_frame = self.visualizer.draw_tracks(frame_data['frame'], [])
                        vis_frame = self.visualizer.add_frame_info(vis_frame, frame_data)
                        vis_frames[cam_id] = vis_frame
                        continue
                    features_dict, extracted_crops_data = self.feature_extractor.extract_features(frame_data['frame'], tracks)
                    tracks_with_features = []
                    track_to_crop_map = {}
                    for i, track in enumerate(tracks):
                        if i in features_dict:
                            track['feature'] = features_dict[i]
                            tracks_with_features.append(track)
                            original_crop_index = -1
                            current_feature_dict_key_count = 0
                            for k_idx, key in enumerate(sorted(features_dict.keys())):
                                if key == i:
                                    original_crop_index = current_feature_dict_key_count
                                    break
                                current_feature_dict_key_count +=1
                            if original_crop_index != -1 and original_crop_index < len(extracted_crops_data):
                                track_to_crop_map[track['track_id']] = extracted_crops_data[original_crop_index]['cv_image']
                            else:
                                track_to_crop_map[track['track_id']] = None
                    if tracks_with_features:
                        updated_tracks_with_gids = self.reid_manager.update(tracks_with_features, frame_data)
                        for updated_track in updated_tracks_with_gids:
                            for original_track in tracks:
                                if original_track['track_id'] == updated_track['track_id']:
                                    original_track['global_id'] = updated_track['global_id']
                                    break
                    # Visualization
                    vis_frame = self.visualizer.draw_tracks(frame_data['frame'], tracks)
                    stats = self.reid_manager.get_stats()
                    vis_frame = self.visualizer.add_frame_info(vis_frame, frame_data, stats)
                    vis_frames[cam_id] = vis_frame
                yield vis_frames
                elapsed = time.time() - start_time
                if elapsed < frame_count / 30:
                    time.sleep(frame_count / 30 - elapsed)
        finally:
            video_handler.release()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Person Re-Identification System")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--videos", type=str, nargs="+", required=True, help="Paths to input video files")
    parser.add_argument("--output", type=str, default="data/output/results.json", help="Path to output file")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = PersonReIDPipeline(args.config)
    
    # Process videos
    pipeline.process_videos(args.videos)
    
    # Save results
    pipeline.save_results(args.output)


if __name__ == "__main__":
    main()