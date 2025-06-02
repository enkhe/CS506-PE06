# CS506-PE06

## Assignment Scripts Overview: Input, Process, Output

This project contains scripts `q1.py` to `q7.py` in the `pe06-machine-learning/` directory, each addressing different stages of a machine learning pipeline for eye detection using Haar cascades.

### Input
- **Images:** JPEG files located in the `img/` directory (e.g., `train1.jpg`, `test1.jpg`, etc.).
- **Haar Cascade XMLs:** `haarcascade_frontalface_default.xml` and `haarcascade_eye.xml` for face and eye detection.
- **Ground Truth Files:** (For evaluation) `.gnd` files in `gnd_files/ground_truth_session/` containing annotated eye coordinates.

### Process
- **q1.py – q2.py:** Detect eyes in images using Haar cascades and save detected coordinates to `.gnd` files in a timestamped session directory.
- **q3.py:** Compare detected coordinates with ground truth, compute matching, and calculate metrics (accuracy, precision, recall, F1-score).
- **q4.py – q6.py:** Perform parameter tuning (grid search) for Haar cascade hyperparameters (`scaleFactor`, `minNeighbors`), visualize performance in 3D, and analyze results.
- **q7.py:** Generate a comprehensive training report and summary of the parameter tuning process.

### Output
- **Detected Coordinates:** `.gnd` files saved in `gnd_files/session_<timestamp>/`, each containing left and right eye coordinates for an image.
- **Performance Metrics:** Printed to the console and optionally saved as part of the report.
- **3D Visualization:** Performance graphs saved as PNG files in the `tuning_graphs/` directory.
- **Reports:** Textual analysis of training and tuning, printed to the console.

See each script's comments and docstrings for further details on usage and workflow.
