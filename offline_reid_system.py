import os
import cv2
import numpy as np
import json
import logging
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import yaml
import torch
import torchreid
from torchvision import transforms
from PIL import Image
import argparse
from datetime import datetime

class OfflineReIDProcessor:
    """Offline re-identification processor for gallery images"""
    
    def __init__(self, config_path, gallery_path=None, logger=None):
        """Initialize the offline ReID processor
        
        Args:
            config_path (str): Path to configuration file
            gallery_path (str, optional): Override gallery path from config
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.gallery_path = gallery_path or self.config['output']['gallery_save_path']
        self.similarity_threshold = self.config['reid'].get('offline_similarity_threshold', 0.75)
        self.min_images_per_person = self.config['reid'].get('min_images_per_person', 3)
        self.max_images_for_comparison = self.config['reid'].get('max_images_for_comparison', 10)
        
        # Initialize feature extractor
        self._initialize_feature_extractor()
        
        # Results storage
        self.merge_results = []
        self.statistics = {}
        
    def _setup_logger(self):
        """Setup logger for offline processing"""
        logger = logging.getLogger("offline_reid")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_feature_extractor(self):
        """Initialize the feature extractor model"""
        try:
            device = self.config['system']['device']
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_path = self.config['reid']['model_path']
            
            self.model = torchreid.utils.FeatureExtractor(
                model_name='osnet_x1_0',
                model_path=model_path,
                device=device
            )
            
            self.transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.logger.info(f"Initialized feature extractor on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feature extractor: {e}")
            raise
    
    def scan_gallery(self):
        """Scan gallery directory and return GID information
        
        Returns:
            dict: Dictionary with GID information
        """
        if not os.path.exists(self.gallery_path):
            self.logger.error(f"Gallery path does not exist: {self.gallery_path}")
            return {}
        
        gid_info = {}
        
        for gid_folder in os.listdir(self.gallery_path):
            if not gid_folder.startswith("GID_"):
                continue
                
            gid = gid_folder.replace("GID_", "")
            folder_path = os.path.join(self.gallery_path, gid_folder)
            
            if not os.path.isdir(folder_path):
                continue
            
            # Get all image files in the folder
            image_files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(folder_path, file))
            
            if len(image_files) >= self.min_images_per_person:
                gid_info[gid] = {
                    'folder_path': folder_path,
                    'image_files': image_files,
                    'num_images': len(image_files)
                }
                
                self.logger.debug(f"Found GID {gid} with {len(image_files)} images")
        
        self.logger.info(f"Found {len(gid_info)} GIDs with sufficient images for processing")
        return gid_info
    
    def extract_features_from_images(self, image_paths, max_images=None):
        """Extract features from a list of image paths
        
        Args:
            image_paths (list): List of image file paths
            max_images (int, optional): Maximum number of images to process
        
        Returns:
            numpy.ndarray: Array of feature vectors
        """
        if max_images:
            # Select diverse images (every nth image)
            step = max(1, len(image_paths) // max_images)
            selected_paths = image_paths[::step][:max_images]
        else:
            selected_paths = image_paths
        
        features = []
        valid_images = 0
        
        for img_path in selected_paths:
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # Apply transformations
                img_tensor = self.transform(img_pil).unsqueeze(0)
                
                # Extract features
                with torch.no_grad():
                    feature = self.model(img_tensor)
                    feature = feature.cpu().numpy().flatten()
                    feature = feature / np.linalg.norm(feature)  # Normalize
                    features.append(feature)
                    valid_images += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to process image {img_path}: {e}")
                continue
        
        if not features:
            return np.array([])
        
        return np.array(features)
    
    def compute_similarity_matrix(self, gid_info):
        """Compute similarity matrix between all GID pairs
        
        Args:
            gid_info (dict): GID information dictionary
        
        Returns:
            dict: Similarity matrix and metadata
        """
        gids = list(gid_info.keys())
        n_gids = len(gids)
        
        # Extract features for each GID
        gid_features = {}
        self.logger.info("Extracting features for all GIDs...")
        
        for i, gid in enumerate(gids):
            self.logger.info(f"Processing GID {gid} ({i+1}/{n_gids})")
            
            features = self.extract_features_from_images(
                gid_info[gid]['image_files'],
                max_images=self.max_images_for_comparison
            )
            
            if len(features) > 0:
                # Use average feature as representative
                avg_feature = np.mean(features, axis=0)
                gid_features[gid] = {
                    'avg_feature': avg_feature,
                    'all_features': features,
                    'num_features': len(features)
                }
            else:
                self.logger.warning(f"No valid features extracted for GID {gid}")
        
        # Compute similarity matrix
        self.logger.info("Computing similarity matrix...")
        similarity_matrix = {}
        similarity_pairs = []
        
        for i, gid1 in enumerate(gids):
            if gid1 not in gid_features:
                continue
                
            for j, gid2 in enumerate(gids[i+1:], i+1):
                if gid2 not in gid_features:
                    continue
                
                # Compute similarity between average features
                feat1 = gid_features[gid1]['avg_feature']
                feat2 = gid_features[gid2]['avg_feature']
                
                similarity = np.dot(feat1, feat2)
                
                # Also compute cross-comparison between all features
                all_similarities = []
                for f1 in gid_features[gid1]['all_features']:
                    for f2 in gid_features[gid2]['all_features']:
                        all_similarities.append(np.dot(f1, f2))
                
                max_similarity = max(all_similarities) if all_similarities else similarity
                avg_cross_similarity = np.mean(all_similarities) if all_similarities else similarity
                
                similarity_matrix[(gid1, gid2)] = {
                    'avg_similarity': similarity,
                    'max_similarity': max_similarity,
                    'avg_cross_similarity': avg_cross_similarity,
                    'num_comparisons': len(all_similarities)
                }
                
                similarity_pairs.append((gid1, gid2, similarity, max_similarity))
        
        # Sort by similarity
        similarity_pairs.sort(key=lambda x: x[3], reverse=True)  # Sort by max similarity
        
        return {
            'similarity_matrix': similarity_matrix,
            'similarity_pairs': similarity_pairs,
            'gid_features': gid_features
        }
    
    def find_merge_candidates(self, similarity_data):
        """Find GID pairs that should be merged
        
        Args:
            similarity_data (dict): Similarity computation results
        
        Returns:
            list: List of merge operations
        """
        merge_candidates = []
        similarity_pairs = similarity_data['similarity_pairs']
        
        # Track which GIDs have already been assigned for merging
        assigned_gids = set()
        
        for gid1, gid2, avg_sim, max_sim in similarity_pairs:
            # Use max similarity for decision making
            if max_sim >= self.similarity_threshold:
                # Skip if either GID is already assigned
                if gid1 in assigned_gids or gid2 in assigned_gids:
                    continue
                
                merge_candidates.append({
                    'source_gid': gid2,  # GID to be merged (typically higher number)
                    'target_gid': gid1,  # GID to merge into (typically lower number)
                    'avg_similarity': avg_sim,
                    'max_similarity': max_sim,
                    'confidence': 'high' if max_sim > 0.85 else 'medium'
                })
                
                # Mark both GIDs as assigned
                assigned_gids.add(gid1)
                assigned_gids.add(gid2)
                
                self.logger.info(f"Merge candidate: GID {gid2} -> GID {gid1} "
                               f"(avg_sim: {avg_sim:.3f}, max_sim: {max_sim:.3f})")
        
        return merge_candidates
    
    def execute_merges(self, merge_candidates, gid_info):
        """Execute the merge operations
        
        Args:
            merge_candidates (list): List of merge operations
            gid_info (dict): GID information dictionary
        
        Returns:
            dict: Merge execution results
        """
        if not merge_candidates:
            self.logger.info("No merges to execute")
            return {'executed': 0, 'failed': 0, 'details': []}
        
        # Create backup directory
        backup_dir = os.path.join(os.path.dirname(self.gallery_path), 
                                 f"gallery_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copytree(self.gallery_path, backup_dir)
        self.logger.info(f"Created backup at: {backup_dir}")
        
        executed = 0
        failed = 0
        merge_details = []
        
        for merge_op in merge_candidates:
            source_gid = merge_op['source_gid']
            target_gid = merge_op['target_gid']
            
            try:
                source_folder = os.path.join(self.gallery_path, f"GID_{source_gid}")
                target_folder = os.path.join(self.gallery_path, f"GID_{target_gid}")
                
                if not os.path.exists(source_folder):
                    self.logger.warning(f"Source folder not found: {source_folder}")
                    failed += 1
                    continue
                
                if not os.path.exists(target_folder):
                    self.logger.warning(f"Target folder not found: {target_folder}")
                    failed += 1
                    continue
                
                # Move all images from source to target
                moved_files = 0
                for file in os.listdir(source_folder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        source_file = os.path.join(source_folder, file)
                        
                        # Rename file to avoid conflicts
                        base_name, ext = os.path.splitext(file)
                        new_name = f"merged_from_GID{source_gid}_{base_name}{ext}"
                        target_file = os.path.join(target_folder, new_name)
                        
                        # Handle name conflicts
                        counter = 1
                        while os.path.exists(target_file):
                            new_name = f"merged_from_GID{source_gid}_{base_name}_{counter}{ext}"
                            target_file = os.path.join(target_folder, new_name)
                            counter += 1
                        
                        shutil.move(source_file, target_file)
                        moved_files += 1
                
                # Remove empty source folder
                if moved_files > 0:
                    os.rmdir(source_folder)
                    executed += 1
                    
                    merge_details.append({
                        'source_gid': source_gid,
                        'target_gid': target_gid,
                        'moved_files': moved_files,
                        'similarity': merge_op['max_similarity'],
                        'confidence': merge_op['confidence']
                    })
                    
                    self.logger.info(f"Successfully merged GID {source_gid} into GID {target_gid} "
                                   f"({moved_files} files moved)")
                else:
                    failed += 1
                    self.logger.warning(f"No files to move for GID {source_gid}")
                    
            except Exception as e:
                self.logger.error(f"Failed to merge GID {source_gid} into {target_gid}: {e}")
                failed += 1
        
        return {
            'executed': executed,
            'failed': failed,
            'details': merge_details,
            'backup_location': backup_dir
        }
    
    def generate_report(self, gid_info, similarity_data, merge_results):
        """Generate a detailed report of the re-identification process
        
        Args:
            gid_info (dict): Original GID information
            similarity_data (dict): Similarity computation results
            merge_results (dict): Merge execution results
        
        Returns:
            dict: Comprehensive report
        """
        # Basic statistics
        original_gids = len(gid_info)
        final_gids = original_gids - merge_results['executed']
        
        # Image distribution
        images_per_gid = [info['num_images'] for info in gid_info.values()]
        
        # High similarity pairs (potential missed merges)
        high_sim_pairs = [
            (gid1, gid2, data['max_similarity']) 
            for (gid1, gid2), data in similarity_data['similarity_matrix'].items()
            if data['max_similarity'] > 0.7 and data['max_similarity'] < self.similarity_threshold
        ]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'gallery_path': self.gallery_path,
            'processing_config': {
                'similarity_threshold': self.similarity_threshold,
                'min_images_per_person': self.min_images_per_person,
                'max_images_for_comparison': self.max_images_for_comparison
            },
            'statistics': {
                'original_gids': original_gids,
                'final_gids': final_gids,
                'gids_merged': merge_results['executed'],
                'merge_failures': merge_results['failed'],
                'total_images': sum(images_per_gid),
                'avg_images_per_gid': np.mean(images_per_gid) if images_per_gid else 0,
                'min_images_per_gid': min(images_per_gid) if images_per_gid else 0,
                'max_images_per_gid': max(images_per_gid) if images_per_gid else 0
            },
            'merge_details': merge_results['details'],
            'high_similarity_pairs': high_sim_pairs[:10],  # Top 10 potential missed merges
            'backup_location': merge_results.get('backup_location')
        }
        
        return report
    
    def save_report(self, report, output_path=None):
        """Save the report to a JSON file
        
        Args:
            report (dict): Report dictionary
            output_path (str, optional): Output file path
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"offline_reid_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Report saved to: {output_path}")
    
    def process_gallery(self, dry_run=False):
        """Main processing function for gallery re-identification
        
        Args:
            dry_run (bool): If True, don't execute merges, just report what would be done
        
        Returns:
            dict: Processing results and report
        """
        self.logger.info("Starting offline gallery re-identification processing...")
        
        # Step 1: Scan gallery
        self.logger.info("Step 1: Scanning gallery...")
        gid_info = self.scan_gallery()
        
        if not gid_info:
            self.logger.warning("No suitable GIDs found for processing")
            return None
        
        # Step 2: Compute similarities
        self.logger.info("Step 2: Computing similarities...")
        similarity_data = self.compute_similarity_matrix(gid_info)
        
        # Step 3: Find merge candidates
        self.logger.info("Step 3: Finding merge candidates...")
        merge_candidates = self.find_merge_candidates(similarity_data)
        
        if not merge_candidates:
            self.logger.info("No merge candidates found")
            merge_results = {'executed': 0, 'failed': 0, 'details': []}
        elif dry_run:
            self.logger.info(f"DRY RUN: Would execute {len(merge_candidates)} merges")
            merge_results = {'executed': 0, 'failed': 0, 'details': [], 'dry_run': True}
        else:
            # Step 4: Execute merges
            self.logger.info(f"Step 4: Executing {len(merge_candidates)} merges...")
            merge_results = self.execute_merges(merge_candidates, gid_info)
        
        # Step 5: Generate report
        self.logger.info("Step 5: Generating report...")
        report = self.generate_report(gid_info, similarity_data, merge_results)
        
        # Print summary
        self.logger.info("=== PROCESSING SUMMARY ===")
        self.logger.info(f"Original GIDs: {report['statistics']['original_gids']}")
        self.logger.info(f"Final GIDs: {report['statistics']['final_gids']}")
        self.logger.info(f"Merges executed: {report['statistics']['gids_merged']}")
        self.logger.info(f"Total images: {report['statistics']['total_images']}")
        
        if merge_results.get('backup_location'):
            self.logger.info(f"Backup created at: {merge_results['backup_location']}")
        
        return report


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Offline Gallery Re-identification Processor')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--gallery', help='Path to gallery directory (overrides config)')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run without executing merges')
    parser.add_argument('--similarity-threshold', type=float, help='Similarity threshold for merging')
    parser.add_argument('--output-report', help='Output path for the report JSON file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize processor
        processor = OfflineReIDProcessor(args.config, gallery_path=args.gallery)
        
        # Override similarity threshold if provided
        if args.similarity_threshold:
            processor.similarity_threshold = args.similarity_threshold
            processor.logger.info(f"Using similarity threshold: {args.similarity_threshold}")
        
        # Process gallery
        report = processor.process_gallery(dry_run=args.dry_run)
        
        if report:
            # Save report
            processor.save_report(report, args.output_report)
            
            print("\n=== FINAL SUMMARY ===")
            print(f"Processing completed successfully!")
            print(f"Original GIDs: {report['statistics']['original_gids']}")
            print(f"Final GIDs: {report['statistics']['final_gids']}")
            print(f"Merges executed: {report['statistics']['gids_merged']}")
            
            if args.dry_run:
                print("(DRY RUN - No actual changes made)")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
