#!/usr/bin/env python
"""
Person Re-Identification System
Main entry point script
"""

import argparse
from src.pipeline import PersonReIDPipeline
from offline_reid_system import OfflineReIDProcessor, main as offline_main

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Person Re-Identification System")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--videos", type=str, nargs="+", required=True, help="Paths to input video files")
    parser.add_argument("--output", type=str, default="data/output/results.json", help="Path to output file")
    parser.add_argument("--offline-reid", action="store_true", help="Run offline ReID processing on the gallery")
    parser.add_argument('--gallery-path', type=str, default=None, help='Path to gallery directory for offline ReID (overrides config)')
    parser.add_argument('--offline-similarity-threshold', type=float, default=None, help='Similarity threshold for offline merging')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run for offline ReID without executing merges')
    parser.add_argument('--output-report', help='Output path for the offline ReID report JSON file')

    args = parser.parse_args()
    
    if args.offline_reid:
        print("Running offline Re-ID system...")
        # For simplicity, we'll call the main function from offline_reid_system.py
        # This assumes offline_reid_system.py is in the same directory or accessible in PYTHONPATH
        # And that its main function can be called with relevant arguments or it handles them internally.
        
        # Construct arguments for offline_main
        offline_args = ['--config', args.config]
        if args.gallery_path:
            offline_args.extend(['--gallery', args.gallery_path])
        if args.offline_similarity_threshold:
            offline_args.extend(['--similarity-threshold', str(args.offline_similarity_threshold)])
        if args.dry_run:
            offline_args.append('--dry-run')
        if args.output_report:
            offline_args.extend(['--output-report', args.output_report])
        
        # Simulate command line arguments for offline_main
        import sys
        original_argv = sys.argv
        sys.argv = [sys.argv[0]] + offline_args
        try:
            offline_main() # Call the main function from offline_reid_system.py
        finally:
            sys.argv = original_argv # Restore original arguments

    else:
        if not args.videos:
            parser.error("The --videos argument is required when not running --offline-reid.")
        # Create pipeline
        pipeline = PersonReIDPipeline(args.config)
        
        # Process videos
        pipeline.process_videos(args.videos)
        
        # Save results
        pipeline.save_results(args.output)


if __name__ == "__main__":
    main()