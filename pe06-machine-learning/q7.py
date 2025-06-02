import cv2
import os
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Explicitly import Axes3D for 3D projection

# --- Configuration & Directory Setup ---
IMG_DIR = "img"
GND_BASE_DIR = "gnd_files" # Base directory for all timestamped gnd folders
HAARCASCADE_DIR = "." # Assuming haarcascade XMLs are in the same directory as the script
GRAPH_OUTPUT_DIR = "tuning_graphs" # Directory to save the 3D graph

CURRENT_SESSION_GND_DIR = None # This will be set dynamically for each run

def setup_directories_and_session():
    """
    Creates the necessary directories if they don't exist, and creates
    a new timestamped session directory for the current run's .gnd files.
    """
    global CURRENT_SESSION_GND_DIR

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(GND_BASE_DIR, exist_ok=True)
    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True) # Create graph output directory

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    CURRENT_SESSION_GND_DIR = os.path.join(GND_BASE_DIR, f"session_{timestamp}")
    os.makedirs(CURRENT_SESSION_GND_DIR, exist_ok=True)
    print(f"Ensured '{IMG_DIR}/' exists.")
    print(f"Current session directory for detected coordinates: '{CURRENT_SESSION_GND_DIR}/'")
    print(f"Graph output directory: '{GRAPH_OUTPUT_DIR}/'")

# --- Utility Functions ---

def load_coordinates(gnd_filepath):
    """
    Loads eye coordinates from a .gnd file.
    Returns (left_eye_coords_tuple, right_eye_coords_tuple) or None if file error.
    """
    try:
        with open(gnd_filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                l_coords_str = lines[0].strip().split(': ')[1]
                r_coords_str = lines[1].strip().split(': ')[1]
                left_eye = tuple(map(int, l_coords_str.split(', ')))
                right_eye = tuple(map(int, r_coords_str.split(', ')))
                return left_eye, right_eye
            else:
                return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  Error reading '{gnd_filepath}': {e}")
        return None

def euclidean_distance(coord1, coord2):
    """Calculates the Euclidean distance between two 2D points."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

# --- Core Detection Logic (from Q2) ---

def detect_eyes_and_save_single_image(image_filepath, face_cascade, eye_cascade, scaleFactor, minNeighbors, save_dir):
    """
    Detects faces and then eyes in a single image with given parameters,
    saves to a .gnd file in the specified directory, and returns detected coordinates.
    """
    img_filename = os.path.basename(image_filepath)
    img = cv2.imread(image_filepath)
    if img is None:
        return None 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(60, 60), # Keep minSize consistent for faces
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    best_left_eye = None
    best_right_eye = None

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True) # Sort by area
        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=(20, 20), # Min eye size
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            current_face_eyes = []
            for (ex, ey, ew, eh) in eyes:
                eye_center_x = fx + ex + ew // 2
                eye_center_y = fy + ey + eh // 2
                current_face_eyes.append((eye_center_x, eye_center_y))

            if len(current_face_eyes) >= 2:
                current_face_eyes.sort(key=lambda coord: coord[0]) # Sort by x-coordinate

                left_candidates = [eye for eye in current_face_eyes if eye[0] < (fx + fw/2)]
                right_candidates = [eye for eye in current_face_eyes if eye[0] >= (fx + fw/2)]
                
                if left_candidates and right_candidates:
                    best_left_eye = left_candidates[0]
                    best_right_eye = right_candidates[-1]
                    break 
    if best_left_eye and best_right_eye:
        gnd_filename = os.path.splitext(img_filename)[0] + '.gnd'
        gnd_filepath = os.path.join(save_dir, gnd_filename)
        with open(gnd_filepath, 'w') as f:
            f.write(f"L: {best_left_eye[0]}, {best_left_eye[1]}\n")
            f.write(f"R: {best_right_eye[0]}, {best_right_eye[1]}\n")
        return best_left_eye, best_right_eye
    return None 

# --- Matching and Metrics (from Q3) ---

def match_faces_to_ground_truth(gt_coords_dict, detected_coords_dict, threshold=20):
    """
    Matches detected faces (based on eye coordinates) to ground truth faces.
    A match occurs if the average distance between corresponding eyes is below a threshold.
    """
    true_positives = 0
    false_negatives_set = set(gt_coords_dict.keys())
    false_positives_set = set(detected_coords_dict.keys())

    for filename, gt_eyes in gt_coords_dict.items():
        if filename in detected_coords_dict:
            det_eyes = detected_coords_dict[filename]

            dist_l = euclidean_distance(gt_eyes[0], det_eyes[0])
            dist_r = euclidean_distance(gt_eyes[1], det_eyes[1])
            avg_dist = (dist_l + dist_r) / 2

            if avg_dist <= threshold:
                true_positives += 1
                false_negatives_set.discard(filename)
                false_positives_set.discard(filename) 

    false_positives = len(false_positives_set)
    false_negatives = len(false_negatives_set)

    return true_positives, false_positives, false_negatives

def calculate_metrics(tp, fp, fn):
    """
    Calculates Accuracy, Precision, Recall, and F1-score.
    """
    actual_positives = tp + fn
    predicted_positives = tp + fp

    accuracy = (tp) / (actual_positives + fp) if (actual_positives + fp) > 0 else 0
    precision = tp / predicted_positives if predicted_positives > 0 else 0
    recall = tp / actual_positives if actual_positives > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score

# --- Question 5: Grid Search Optimization and 3D Visualization ---

def perform_grid_search_and_visualize(
    image_filenames, 
    gt_coordinates_by_image, 
    face_cascade, 
    eye_cascade,
    scale_factor_range, 
    min_neighbors_range,
    matching_threshold=20 # Default matching threshold for metrics
):
    """
    Performs a grid search over scaleFactor and minNeighbors, calculates metrics,
    and visualizes the F1-score performance in a 3D graph.
    """
    best_f1 = -1
    best_params = {}
    all_results = []

    # Prepare data storage for 3D plot
    sf_mesh, mn_mesh = np.meshgrid(scale_factor_range, min_neighbors_range)
    f1_scores_matrix = np.zeros(sf_mesh.shape)

    print("\n--- Starting Grid Search Optimization ---")
    print("Evaluating all parameter combinations...")

    for i, sf in enumerate(scale_factor_range):
        for j, mn in enumerate(min_neighbors_range):
            sf_val = float(sf)
            mn_val = int(mn)

            # Create a temporary directory for detections of this specific parameter set
            temp_detection_dir = os.path.join(CURRENT_SESSION_GND_DIR, f"sf{sf_val:.2f}_mn{mn_val}")
            os.makedirs(temp_detection_dir, exist_ok=True) # Ensure it's created
            
            # print(f"  Testing sf={sf_val:.2f}, mn={mn_val}...") # Uncomment for very verbose output

            detected_coords_for_params = {}
            for filename in image_filenames:
                image_filepath = os.path.join(IMG_DIR, filename)
                
                detected_result = detect_eyes_and_save_single_image(
                    image_filepath, face_cascade, eye_cascade, sf_val, mn_val, temp_detection_dir
                )
                if detected_result:
                    detected_coords_for_params[filename] = detected_result

            # Calculate metrics for this parameter set
            tp, fp, fn = match_faces_to_ground_truth(
                gt_coordinates_by_image, 
                detected_coords_for_params, 
                threshold=matching_threshold
            )
            accuracy, precision, recall, f1_score = calculate_metrics(tp, fp, fn)

            f1_scores_matrix[j, i] = f1_score # Store F1-score for plotting

            current_params_results = {
                'scaleFactor': sf_val,
                'minNeighbors': mn_val,
                'TP': tp, 'FP': fp, 'FN': fn,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score
            }
            all_results.append(current_params_results)

            if f1_score > best_f1:
                best_f1 = f1_score
                best_params = current_params_results # Update best params
    
    # --- Print All Results ---
    print("\n--- All Grid Search Results (Sorted by F1-Score) ---")
    all_results_sorted = sorted(all_results, key=lambda x: x['F1-Score'], reverse=True)
    for res in all_results_sorted:
        print(f"sf={res['scaleFactor']:.2f}, mn={res['minNeighbors']}: F1={res['F1-Score']:.4f}, P={res['Precision']:.4f}, R={res['Recall']:.4f}, Acc={res['Accuracy']:.4f} (TP={res['TP']}, FP={res['FP']}, FN={res['FN']})")

    # --- Print Best Parameters ---
    print("\n--- Best Performing Parameters from Grid Search ---")
    if best_params:
        print(f"Optimal F1-Score: {best_params['F1-Score']:.4f}")
        print(f"  Parameters: scaleFactor={best_params['scaleFactor']:.2f}, minNeighbors={best_params['minNeighbors']}")
        print(f"  Accuracy: {best_params['Accuracy']:.4f}")
        print(f"  Precision: {best_params['Precision']:.4f}")
        print(f"  Recall: {best_params['Recall']:.4f}")
        print(f"  TP: {best_params['TP']}, FP: {best_params['FP']}, FN: {best_params['FN']}")
    else:
        print("No successful detections found for any parameter set during grid search.")

    # --- 3D Visualization ---
    fig = plt.figure(figsize=(14, 12), dpi=300) 
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(sf_mesh, mn_mesh, f1_scores_matrix, cmap='viridis', 
                           edgecolor='none', alpha=0.8)

    if best_params:
        ax.scatter(best_params['scaleFactor'], best_params['minNeighbors'], best_params['F1-Score'], 
                   color='red', marker='o', s=150, label=f"Best F1: {best_params['F1-Score']:.2f}", depthshade=True, zorder=5)

    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Min Neighbors')
    ax.set_zlabel('F1-Score')
    ax.set_title('Haar Cascade Tuning: F1-Score Performance Landscape')

    fig.colorbar(surf, shrink=0.5, aspect=5, label='F1-Score')
    ax.view_init(elev=30, azim=135) 

    graph_filename = f"3D_F1_Performance_GridSearch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    graph_filepath = os.path.join(GRAPH_OUTPUT_DIR, graph_filename)
    plt.savefig(graph_filepath, bbox_inches='tight')
    print(f"\n3D performance graph saved to: '{graph_filepath}'")
    plt.close(fig)

    return {"all_results": all_results, "best_params": best_params}


# --- Question 6: Training Process Analysis Report ---

def generate_training_report(tuning_results, scale_factor_range, min_neighbors_range):
    """
    Generates a textual report analyzing the training process and performance.
    """
    all_results = tuning_results["all_results"]
    best_params = tuning_results["best_params"]

    print("\n" + "="*60)
    print("                 TRAINING PROCESS ANALYSIS REPORT                 ")
    print("="*60)

    print("\n1. Analysis of Training Process:")
    print("--------------------------------")
    print("The training process involved a systematic exploration of the Haar Cascade classifier's hyperparameters, specifically `scaleFactor` and `minNeighbors`, using a Grid Search optimization strategy.")
    print(f"- `scaleFactor` was varied from {scale_factor_range.min():.2f} to {scale_factor_range.max():.2f} with a step of {scale_factor_range[1]-scale_factor_range[0]:.2f}.")
    print(f"- `minNeighbors` was varied from {min_neighbors_range.min()} to {min_neighbors_range.max()} with a step of {min_neighbors_range[1]-min_neighbors_range[0]:.0f}.")
    print(f"Each combination of these parameters was tested against a predefined set of {len(image_filenames_to_process)} images, comparing the automatically detected eye coordinates against a set of ground truth labels (simulated in this exercise).")
    print("The performance for each parameter set was evaluated using standard classification metrics: True Positives (TP), False Positives (FP), False Negatives (FN), Accuracy, Precision, Recall, and F1-Score. The F1-Score was chosen as the primary metric for optimization, as it provides a balanced measure of both precision and recall.")

    print("\n2. Performance Interpretation: Peaks and Valleys")
    print("------------------------------------------------")
    print("The 3D surface plot generated during the tuning process visually represents the F1-Score across the parameter space.")
    print("On this plot:")
    print("- **Peaks (higher F1-Score values, typically yellow/lighter colors in the 'viridis' colormap)** indicate parameter combinations that resulted in **better overall detection performance**. These are regions where the classifier achieved a good balance between identifying correct instances (high recall) and minimizing incorrect identifications (high precision).")
    print("- **Valleys (lower F1-Score values, typically purple/darker colors)** represent parameter combinations that led to **worse detection performance**. This could be due to `scaleFactor` values that are too aggressive (missing smaller features) or too fine (leading to excessive false positives), or `minNeighbors` values that are either too lenient (many false positives) or too strict (missing true detections).")
    print("By observing the contours and gradients of the surface, one can identify optimal regions and understand the sensitivity of the model to changes in these hyperparameters.")
    
    print("\n3. Best Parameter Set Found:")
    print("----------------------------")
    if best_params:
        print(f"Based on the Grid Search, the optimal parameter set achieving the highest F1-Score is:")
        print(f"  - `scaleFactor`: {best_params['scaleFactor']:.2f}")
        print(f"  - `minNeighbors`: {best_params['minNeighbors']}")
        print(f"  - **Optimal F1-Score**: {best_params['F1-Score']:.4f}")
        print("\n4. Performance Metrics for the Best Parameter Set:")
        print("--------------------------------------------------")
        print(f"  - True Positives (TP): {best_params['TP']}")
        print(f"  - False Positives (FP): {best_params['FP']}")
        print(f"  - False Negatives (FN): {best_params['FN']}")
        print(f"  - Accuracy: {best_params['Accuracy']:.4f}")
        print(f"  - Precision: {best_params['Precision']:.4f}")
        print(f"  - Recall: {best_params['Recall']:.4f}")
    else:
        print("No optimal parameter set could be determined as no successful detections were recorded.")

    print("\n" + "="*60)
    print("                         REPORT END                         ")
    print("="*60)


# --- Main Program Execution Start ---
if __name__ == "__main__":
    # Ensure this variable is accessible globally or passed correctly
    image_filenames_to_process = [
        "train1.jpg", "train2.jpg", "train3.jpg", "train4.jpg", "train5.jpg",
        "sample.jpg", "test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg"
    ]
    
    # Initialize directories for the session
    setup_directories_and_session()

    # --- Prepare Ground Truth Data (simulated as in previous Qs) ---
    ground_truth_coordinates_by_image = {
        "train1.jpg": ((238, 357), (421, 357)),
        "train2.jpg": ((100, 150), (200, 150)), 
        "train3.jpg": ((120, 180), (220, 180)), 
        "train4.jpg": ((110, 160), (210, 160)), 
        "train5.jpg": ((130, 190), (230, 190)), 
        "sample.jpg": ((299, 362), (465, 362)), 
        "test1.jpg": ((270, 390), (420, 390)), 
        "test2.jpg": ((300, 250), (400, 250)), 
        "test3.jpg": ((250, 270), (450, 270)), 
        "test4.jpg": ((300, 200), (450, 200)), 
        "test5.jpg": ((370, 370), (510, 370))  
    }

    # --- Load Haar Cascades ---
    face_cascade_path = os.path.join(HAARCASCADE_DIR, "haarcascade_frontalface_default.xml")
    eye_cascade_path = os.path.join(HAARCASCADE_DIR, "haarcascade_eye.xml")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty():
        print(f"Error: Face cascade XML not loaded from {face_cascade_path}. Check path and file integrity.")
        exit()
    if eye_cascade.empty():
        print(f"Error: Eye cascade XML not loaded from {eye_cascade_path}. Check path and file integrity.")
        exit()

    # --- Define Parameter Ranges for Grid Search ---
    scale_factor_values = np.arange(1.01, 2.01, 0.05).round(2) 
    min_neighbors_values = np.arange(1, 11, 1) 

    # --- Perform Grid Search and Get Tuning Results ---
    # This function now returns the results directly for the report.
    tuning_results = perform_grid_search_and_visualize(
        image_filenames_to_process,
        ground_truth_coordinates_by_image,
        face_cascade,
        eye_cascade,
        scale_factor_values,
        min_neighbors_values,
        matching_threshold=20 # Threshold for matching GT to detected
    )

    # --- Generate and Print Report (Question 6) ---
    generate_training_report(tuning_results, scale_factor_values, min_neighbors_values)

    print("\nProgram finished.")