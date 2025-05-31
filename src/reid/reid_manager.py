import numpy as np
import logging
import time
from collections import defaultdict

class Person:
    """Person class for re-identification gallery"""
    def __init__(self, global_id, feature, cam_id, track_id, config):
        """Initialize a new person
        
        Args:
            global_id (int): Global person ID
            feature (numpy.ndarray): Feature vector
            cam_id (int): Camera ID
            track_id (int): Track ID
            config (dict): Configuration dictionary
        """
        self.global_id = global_id
        self.features = [feature]
        self.cam_ids = [cam_id]
        self.track_ids = [track_id]
        self.last_seen = time.time()
        self.feature_count = 1
        self.config = config
    
    def update(self, feature, cam_id, track_id):
        """Update person with new feature
        
        Args:
            feature (numpy.ndarray): Feature vector
            cam_id (int): Camera ID
            track_id (int): Track ID
        """
        self.features.append(feature)
        self.cam_ids.append(cam_id)
        self.track_ids.append(track_id)
        self.last_seen = time.time()
        self.feature_count += 1
        
        # Limit number of features and implement feature averaging for stability
        max_features = self.config.get('reid', {}).get('max_features_per_person', 10) 
        if len(self.features) > max_features:
            # Simple moving average: remove oldest, new one already appended
            self.features.pop(0)
            self.cam_ids.pop(0)
            self.track_ids.pop(0)
    
    def get_average_feature(self):
        """Get average feature vector
        
        Returns:
            numpy.ndarray: Average feature vector
        """
        return np.mean(self.features, axis=0)


class ReIDManager:
    """Re-identification manager for cross-camera tracking"""
    
    def __init__(self, config, logger=None):
        """Initialize the re-identification manager
        
        Args:
            config (dict): Configuration dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger("person_reid")
        
        # Extract configuration
        self.similarity_threshold = config['reid']['similarity_threshold']
        self.gallery_size = config['reid']['gallery_size']
        
        # Initialize gallery
        self.gallery = {}  # {global_id: Person}
        self.next_global_id = 1
        
        # Track to global ID mapping
        self.track_to_global = {}  # {(cam_id, track_id): global_id}
        
        # Lost tracks history for recovery
        self.lost_tracks = defaultdict(list)  # {cam_id: [(track_id, global_id, last_feature, last_seen_frame)]}

        # For merging global IDs
        self.merge_threshold = config['reid'].get('merge_threshold', self.similarity_threshold + 0.1) # Stricter for merging
        self.last_merge_check_time = time.time()
        self.merge_check_interval = config['reid'].get('merge_check_interval', 60) # seconds
    
    def _compute_similarity(self, feature1, feature2):
        """Compute cosine similarity between feature vectors
        
        Args:
            feature1 (numpy.ndarray): First feature vector
            feature2 (numpy.ndarray): Second feature vector
        
        Returns:
            float: Cosine similarity score
        """
        return np.dot(feature1, feature2)
    
    def _find_best_match(self, feature, cam_id):
        """Find best matching person in gallery
        
        Args:
            feature (numpy.ndarray): Feature vector
            cam_id (int): Camera ID
        
        Returns:
            tuple: (global_id, similarity) or (None, 0) if no match
        """
        best_similarity = 0
        best_global_id = None
        
        for global_id, person in self.gallery.items():
            # Skip if same camera and recently seen (likely same track)
            if cam_id in person.cam_ids and (time.time() - person.last_seen) < 5.0:
                continue
                
            # Compute similarity with average feature
            avg_feature = person.get_average_feature()
            similarity = self._compute_similarity(feature, avg_feature)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_global_id = global_id
        
        return best_global_id, best_similarity
    
    def update(self, tracks_with_features, frame_data):
        """Update gallery with new tracks and features
        
        Args:
            tracks_with_features (list): List of tracks with features
            frame_data (dict): Frame data dictionary
        
        Returns:
            dict: Updated tracks with global IDs
        """
        cam_id = frame_data['cam_id']
        frame_id = frame_data['frame_id']
        
        updated_tracks = []
        
        for track in tracks_with_features:
            track_id = track['track_id']
            feature = track['feature']
            
            # Check if track already has a global ID
            track_key = (cam_id, track_id)
            if track_key in self.track_to_global:
                global_id = self.track_to_global[track_key]
                
                # Update person in gallery
                if global_id in self.gallery:
                    self.gallery[global_id].update(feature, cam_id, track_id)
                    
                # Add global ID to track
                track['global_id'] = global_id
                updated_tracks.append(track)
                continue
            
            # Find best match in gallery
            best_global_id, similarity = self._find_best_match(feature, cam_id)
            
            # If good match found, assign existing global ID
            if best_global_id is not None and similarity >= self.similarity_threshold:
                global_id = best_global_id
                self.gallery[global_id].update(feature, cam_id, track_id)
                self.track_to_global[track_key] = global_id
                
                self.logger.debug(f"Matched track {track_id} from camera {cam_id} "
                                 f"to existing person {global_id} with similarity {similarity:.3f}")
            
            # Otherwise, create new person
            else:
                global_id = self.next_global_id
                self.next_global_id += 1
                self.gallery[global_id] = Person(global_id, feature, cam_id, track_id, self.config)
                self.track_to_global[track_key] = global_id
                
                self.logger.debug(f"Created new person {global_id} for track {track_id} "
                                 f"from camera {cam_id}")
            
            # Add global ID to track
            track['global_id'] = global_id
            updated_tracks.append(track)
        
        # Prune gallery if needed
        if len(self.gallery) > self.gallery_size:
            self._prune_gallery()

        # Periodically check for merging global IDs
        current_time = time.time()
        if current_time - self.last_merge_check_time > self.merge_check_interval:
            self._merge_global_ids()
            self.last_merge_check_time = current_time
            
        return updated_tracks
    
    def _prune_gallery(self):
        """Prune gallery to keep it within size limit"""
        # Sort by last seen time
        sorted_gallery = sorted(
            self.gallery.items(),
            key=lambda x: x[1].last_seen
        )
        
        # Remove oldest entries
        num_to_remove = len(self.gallery) - self.gallery_size
        for i in range(num_to_remove):
            global_id, _ = sorted_gallery[i]
            del self.gallery[global_id]
            
            # Remove from track_to_global
            keys_to_remove = []
            for key, gid in self.track_to_global.items():
                if gid == global_id:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.track_to_global[key]
        
        self.logger.debug(f"Pruned {num_to_remove} entries from gallery")

    def _merge_global_ids(self):
        """Post-processing step to merge global IDs that likely belong to the same person."""
        if len(self.gallery) < 2:
            return

        self.logger.info("Performing global ID merge check.")
        merged_something = False
        gallery_items = list(self.gallery.items())
        # Iterate through all unique pairs of persons in the gallery
        for i in range(len(gallery_items)):
            gid1, person1 = gallery_items[i]
            if gid1 not in self.gallery: # May have been merged already
                continue
            avg_feat1 = person1.get_average_feature()

            for j in range(i + 1, len(gallery_items)):
                gid2, person2 = gallery_items[j]
                if gid2 not in self.gallery: # May have been merged already
                    continue
                avg_feat2 = person2.get_average_feature()

                similarity = self._compute_similarity(avg_feat1, avg_feat2)

                if similarity >= self.merge_threshold:
                    self.logger.info(f"Merging global ID {gid2} into {gid1} (similarity: {similarity:.3f})")
                    # Merge person2 into person1
                    for feat, cam, trk_id in zip(person2.features, person2.cam_ids, person2.track_ids):
                        person1.update(feat, cam, trk_id)
                    
                    # Update track_to_global mapping for all tracks associated with gid2
                    for track_key, global_id_val in list(self.track_to_global.items()):
                        if global_id_val == gid2:
                            self.track_to_global[track_key] = gid1
                    
                    # Remove person2 from gallery
                    del self.gallery[gid2]
                    merged_something = True
                    # Break from inner loop as person2 is gone, and person1's feature avg changed
                    break 
            if merged_something and i < len(gallery_items) -1: # If a merge happened, re-evaluate person1 with others
                 # Restart outer loop with current person1 as its features changed
                 # This is a simplification; a more robust approach might re-evaluate all pairs or use a graph-based method
                 self.logger.debug(f"Re-evaluating merges for {gid1} due to recent merge.")
                 # For simplicity, we just log and continue. A full re-evaluation could be complex.
                 # Consider re-fetching gallery_items or a more sophisticated loop structure if many merges are expected
                 # and precise cascading merges are critical.
                 pass # gallery_items is a snapshot, so this is okay for one pass.

        if merged_something:
            self.logger.info("Finished global ID merge check. Some IDs were merged.")
        else:
            self.logger.info("Finished global ID merge check. No IDs were merged.")

    def get_stats(self):
        """Get statistics about the gallery
        
        Returns:
            dict: Gallery statistics
        """
        # Count persons by camera
        persons_by_camera = defaultdict(int)
        for person in self.gallery.values():
            for cam_id in set(person.cam_ids):
                persons_by_camera[cam_id] += 1
        
        # Count cross-camera matches
        cross_camera_matches = 0
        for person in self.gallery.values():
            if len(set(person.cam_ids)) > 1:
                cross_camera_matches += 1
        
        return {
            'total_persons': len(self.gallery),
            'persons_by_camera': dict(persons_by_camera),
            'cross_camera_matches': cross_camera_matches
        }