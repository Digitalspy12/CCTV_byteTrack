# Person Detection, Tracking & Re-Identification System

A comprehensive system for detecting, tracking, and re-identifying people across multiple camera views.

## Features

- Person detection using YOLOv8
- Multi-object tracking based on ByteTrack principles
- Person re-identification using OSNet
- Cross-camera identity matching
- Real-time visualization
- Support for multiple video sources

## System Architecture

```
Video Input → Person Detection → Object Tracking → Feature Extraction → Re-ID Matching → Global ID Assignment → Output
```

## Requirements

- Python 3.8+
- PyTorch 1.12.0+
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd person-reid-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download pre-trained models:
   ```
   mkdir -p models
   # YOLOv8 will be downloaded automatically
   # Download OSNet model
   wget -P models/ https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_market1501.pth
   ```
## TO RUN USING STREAMLIT 
'''
streamlit run app.py
'''
## Usage

### Basic Usage

Run the system with two video files:

```
python src/pipeline.py --videos path/to/video1.mp4 path/to/video2.mp4
```

### Advanced Options

```
python src/pipeline.py --config configs/config.yaml --videos path/to/video1.mp4 path/to/video2.mp4 --output data/output/results.json
```

### Configuration

The system can be configured through the `configs/config.yaml` file. Key configuration options include:

- `system`: General system settings (device, batch size, etc.)
- `video`: Video input settings (frame skip, resize factor, etc.)
- `detection`: YOLOv8 detection settings
- `tracking`: ByteTrack tracking settings
- `reid`: OSNet re-identification settings
- `output`: Output and visualization settings

## System Components

### 1. Video Handler

Manages multiple video inputs and synchronizes frames across cameras.

### 2. Person Detector

Uses YOLOv8 to detect people in each frame with high accuracy.

### 3. Multi-Object Tracker

Tracks detected people across frames using IoU-based matching and motion prediction.

### 4. Feature Extractor

Extracts discriminative features from person crops using OSNet.

### 5. Re-ID Manager

Matches people across cameras using feature similarity and maintains a gallery of known identities.

### 6. Visualizer

Provides real-time visualization of tracking and re-identification results.

## Output

The system generates the following outputs:

- Real-time visualization with bounding boxes and IDs
- Video files with tracking and re-identification results
- JSON file with statistics and results

## License

[MIT License](LICENSE)

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) 
