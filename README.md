# Parking Slot Detection using CNN & OpenCV

This project detects parking slot occupancy in images or videos using a Convolutional Neural Network (CNN) and OpenCV. It provides a tool to create parking slot masks, runs inference to classify each slot as empty or occupied, and outputs results in a CSV file.

## Features

- Detects and classifies parking slots as empty or occupied.
- Supports both image and video input.
- Interactive tool to define parking slot regions.
- Outputs results to a CSV file.
- Uses a trained CNN model for robust classification.

## Project Structure

- `main.py` — Main script for running detection and generating results.
- `util.py` — Utility functions for slot extraction and prediction.
- `create_parking_mask.py` — Tool to interactively define parking slots and generate a mask.
- `cnn_parking_model.h5` — Pre-trained CNN model.
- `mask_1920_1080.png` — Example mask file for slot locations.
- `input.jpg` — Example input image.
- `output/parking_status.csv` — Output CSV with slot statistics.
- `requirements.txt` — Python dependencies.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Create Parking Slot Mask

Run the mask creation tool and follow the on-screen instructions to select parking slots:

```bash
python create_parking_mask.py
```
- Click to draw rectangles for each slot.
- Press `q` to finish and save.

This generates:
- `parking_spots.txt` — Slot coordinates.
- `mask_1920_1080.png` — Binary mask image.

### 2. Run Detection

To detect parking slot status on an image:

```bash
python main.py
```
- By default, uses `mask_1920_1080.png` and `input.jpg`.
- Modify `main.py` or pass arguments to use other files.

Results are shown visually and saved to `output/parking_status.csv`.

### 3. Video Support

If you provide a video file as input, the script will process frames and display real-time results.

## Output

- Visual display with colored rectangles (green: available, red: occupied).
- CSV file with total, occupied, and available slot counts.

## Model

- The CNN model (`cnn_parking_model.h5`) is trained to classify cropped parking slot images as empty or occupied.
- You can retrain or replace the model as needed.

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions.

## Acknowledgements

- Built with OpenCV and TensorFlow/Keras.
- For academic and demonstration purposes.
