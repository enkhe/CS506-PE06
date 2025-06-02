import os
import math
import datetime
import cv2

# --- Configuration ---
IMG_DIR = "img"
GND_BASE_DIR = "gnd_files" # Where all timestamped sessions are stored
HAARCASCADE_DIR = "." # Assuming haarcascade XMLs are here

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "haarcascade_eye.xml"


# --- Core Detection Logic ---
def detect_eyes_and_save(image_filepath, face_cascade, eye_cascade, session_dir=None):
    """
    Detects faces and then eyes within faces, estimates eye centers,
    saves to a .gnd file, and returns the detected coordinates for console output.
    Prioritizes faces by size and eyes by x-coordinate.
    """
    img_filename = os.path.basename(image_filepath)
    img = cv2.imread(image_filepath)
    if img is None:
        print(f"Error: Could not load image {image_filepath}. Skipping.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60), # Increased minSize for better face detection
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        print(f"No faces detected in {img_filename}. Skipping eye detection for this image.")
        return None

    # Sort faces by size (area) in descending order to prioritize larger faces
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    best_left_eye = None
    best_right_eye = None

    for (fx, fy, fw, fh) in faces:
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20), # Min eye size
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        current_face_eyes = []
        for (ex, ey, ew, eh) in eyes:
            # Calculate absolute eye center coordinates
            eye_center_x = fx + ex + ew // 2
            eye_center_y = fy + ey + eh // 2
            current_face_eyes.append((eye_center_x, eye_center_y))

        # If we found at least two eyes in this face, try to assign left/right
        if len(current_face_eyes) >= 2:
            current_face_eyes.sort(key=lambda coord: coord[0]) # Sort by x-coordinate

            # Simple heuristic: first is left, last is right.
            # Could add more sophisticated checks (e.g., vertical alignment within face ROI)
            # but for basic Haar, this is common.
            left_candidates = [eye for eye in current_face_eyes if eye[0] < (fx + fw/2)]
            right_candidates = [eye for eye in current_face_eyes if eye[0] >= (fx + fw/2)]
            
            if left_candidates and right_candidates:
                # Pick the closest eye to the center of its half of the face
                # or simply the leftmost/rightmost in their respective halves
                left_eye_candidate = left_candidates[0] # Leftmost in left half
                right_eye_candidate = right_candidates[-1] # Rightmost in right half

                # Only update if this is the first successful pair found or better in some way (e.g. larger face)
                # For simplicity, we just take the first valid pair found in the largest face.
                best_left_eye = left_eye_candidate
                best_right_eye = right_eye_candidate
                break # Found eyes in the largest face, move on.

    if best_left_eye and best_right_eye:
        if session_dir is not None:
            save_coordinates_to_file(session_dir, img_filename, best_left_eye, best_right_eye)
        return {
            'image': img_filename,
            'left_eye': best_left_eye,
            'right_eye': best_right_eye
        }
    else:
        print(f"  Could not reliably detect a pair of left/right eyes in {img_filename}.")
        return None


def save_coordinates_to_file(session_dir, filename, left_eye, right_eye):
    """
    Saves the provided eye coordinates to a .gnd file in the specified session_dir.
    """
    gnd_filename = os.path.splitext(filename)[0] + '.gnd'
    gnd_filepath = os.path.join(session_dir, gnd_filename)
    with open(gnd_filepath, 'w') as f:
        f.write(f"L: {left_eye[0]}, {left_eye[1]}\n")
        f.write(f"R: {right_eye[0]}, {right_eye[1]}\n")
    print(f"  Coordinates saved to {gnd_filepath}")


def find_latest_session_dir():
    """Finds the most recent timestamped session directory in GND_BASE_DIR."""
    full_path_base_dir = os.path.abspath(GND_BASE_DIR)
    
    if not os.path.exists(full_path_base_dir):
        print(f"Error: Base GND directory '{full_path_base_dir}' does not exist.")
        return None

    session_dirs = [d for d in os.listdir(full_path_base_dir) if os.path.isdir(os.path.join(full_path_base_dir, d)) and d.startswith("session_")]
    if not session_dirs:
        print(f"No session directories found in '{full_path_base_dir}'.")
        return None

    # Sort by name (which includes timestamp) to get the latest
    latest_session_dir = sorted(session_dirs, reverse=True)[0]
    return os.path.join(full_path_base_dir, latest_session_dir)

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
                print(f"  Warning: '{gnd_filepath}' has fewer than 2 lines. Skipping.")
                return None
    except FileNotFoundError:
        print(f"  Warning: '{gnd_filepath}' not found.")
        return None
    except Exception as e:
        print(f"  Error reading '{gnd_filepath}': {e}")
        return None

# --- Question 3: Matching and Metrics ---

def euclidean_distance(coord1, coord2):
    """Calculates the Euclidean distance between two 2D points."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def match_faces_to_ground_truth(gt_coords_dict, detected_coords_dict, threshold=20):
    """
    Matches detected faces (based on eye coordinates) to ground truth faces.
    A match occurs if the average distance between corresponding eyes is below a threshold.

    Args:
        gt_coords_dict (dict): Dictionary mapping image filenames to (left_eye_gt, right_eye_gt) tuples.
        detected_coords_dict (dict): Dictionary mapping image filenames to (left_eye_det, right_eye_det) tuples.
        threshold (int): Maximum allowed average Euclidean distance for a match.

    Returns:
        tuple: (
            matched_pairs (list of tuples): List of (filename, gt_coords, detected_coords, avg_dist) for matches.
            false_positives (list of filenames): Images where detection occurred but no GT match.
            false_negatives (list of filenames): Images where GT exists but no detection match.
        )
    """
    matched_pairs = []
    
    # Initialize all GT images as potential false negatives
    false_negatives_set = set(gt_coords_dict.keys())
    
    # Initialize all detected images as potential false positives
    false_positives_set = set(detected_coords_dict.keys())

    for filename, gt_eyes in gt_coords_dict.items():
        if filename in detected_coords_dict:
            det_eyes = detected_coords_dict[filename]

            dist_l = euclidean_distance(gt_eyes[0], det_eyes[0])
            dist_r = euclidean_distance(gt_eyes[1], det_eyes[1])
            avg_dist = (dist_l + dist_r) / 2

            if avg_dist <= threshold:
                matched_pairs.append({
                    'image': filename,
                    'gt_left': gt_eyes[0],
                    'gt_right': gt_eyes[1],
                    'det_left': det_eyes[0],
                    'det_right': det_eyes[1],
                    'avg_dist': avg_dist
                })
                # If a match is found, remove from both false sets
                false_negatives_set.discard(filename)
                false_positives_set.discard(filename)
            # else: no match for this specific detected pair, so it remains a potential false positive for detection,
            # and the GT remains a potential false negative.

    # Filter false positives and false negatives based on the actual matches found
    false_positives = list(false_positives_set)
    false_negatives = list(false_negatives_set)

    return matched_pairs, false_positives, false_negatives

def calculate_and_print_metrics(matched_pairs, false_positives, false_negatives):
    """
    Calculates and prints detection metrics (Accuracy, Precision, Recall, F1-score).
    """
    true_positives = len(matched_pairs)
    
    # Total actual positives (GT faces)
    actual_positives = true_positives + len(false_negatives) 
    
    # Total detected positives (matched + false positives)
    predicted_positives = true_positives + len(false_positives) 

    # Metrics calculation
    accuracy = (true_positives) / (actual_positives + len(false_positives)) if (actual_positives + len(false_positives)) > 0 else 0
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Detection Metrics ---")
    print(f"True Positives (TP): {true_positives}")
    print(f"False Positives (FP): {len(false_positives)}")
    print(f"False Negatives (FN): {len(false_negatives)}")
    print(f"Total Actual Positives (GT): {actual_positives}")
    print(f"Total Predicted Positives (Detected): {predicted_positives}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

    if false_positives:
        print("\nFalse Positives (Detected but no GT match):")
        for fp_img in false_positives:
            print(f"  - {fp_img}")
    if false_negatives:
        print("\nFalse Negatives (GT but no Detection match):")
        for fn_img in false_negatives:
            print(f"  - {fn_img}")

# --- Main Execution ---
if __name__ == "__main__":
    # Removed setup_directories() call since it is not defined and not needed for this script.

    # Define a list of image filenames for both GT and detected
    image_filenames = [
        "train1.jpg", "train2.jpg", "train3.jpg", "train4.jpg", "train5.jpg",
        "sample.jpg", "test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg"
    ]

    # --- Step 1: Manual GT Labeling (as provided in initial Q1) ---
    # This dictionary represents the "ground truth" labels.
    # In a real scenario, these would be created manually by human annotators.
    # I'm using coordinates from the problem description for train1.jpg and placeholders for others.
    
    # You would typically have these GT files pre-made and just load them.
    # For this script, we'll simulate GT directly from a dictionary.
    
    # NOTE: These are example GT coordinates. You should replace placeholders
    # with actual human-annotated coordinates if you have them.
    ground_truth_coordinates_by_image = {
        "train1.jpg": ((238, 357), (421, 357)),
        "train2.jpg": ((100, 150), (200, 150)), # Placeholder GT
        "train3.jpg": ((120, 180), (220, 180)), # Placeholder GT
        "train4.jpg": ((110, 160), (210, 160)), # Placeholder GT
        "train5.jpg": ((130, 190), (230, 190)), # Placeholder GT
        "sample.jpg": ((299, 362), (465, 362)), # Placeholder GT
        "test1.jpg": ((270, 390), (420, 390)), # Placeholder GT
        "test2.jpg": ((300, 250), (400, 250)), # Placeholder GT
        "test3.jpg": ((250, 270), (450, 270)), # Placeholder GT
        "test4.jpg": ((300, 200), (450, 200)), # Placeholder GT
        "test5.jpg": ((370, 370), (510, 370))  # Placeholder GT
    }

    # Simulate saving GT to files in a "ground_truth_session" folder
    # This is to fulfill the structure for `load_coordinates`
    GT_SESSION_DIR = os.path.join(GND_BASE_DIR, "ground_truth_session")
    os.makedirs(GT_SESSION_DIR, exist_ok=True)
    print(f"\nSimulating saving Ground Truth to: '{GT_SESSION_DIR}/'")
    for filename, coords in ground_truth_coordinates_by_image.items():
        gt_filepath = os.path.join(GT_SESSION_DIR, os.path.splitext(filename)[0] + '.gnd')
        with open(gt_filepath, 'w') as f:
            f.write(f"L: {coords[0][0]}, {coords[0][1]}\n")
            f.write(f"R: {coords[1][0]}, {coords[1][1]}\n")
        # print(f"  Saved GT for {filename}") # Uncomment for verbose output

    # --- Step 2: Automated Detection using Haar Cascades ---
    # This part leverages the previous script's logic to generate 'detected' files.
    # It requires haarcascade XMLs and OpenCV.
    
    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(os.path.join(HAARCASCADE_DIR, "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(os.path.join(HAARCASCADE_DIR, "haarcascade_eye.xml"))

    if face_cascade.empty():
        print(f"Error: Face cascade XML not loaded from {FACE_CASCADE_PATH}. Check path.")
        exit()
    if eye_cascade.empty():
        print(f"Error: Eye cascade XML not loaded from {EYE_CASCADE_PATH}. Check path.")
        exit()

    detected_coordinates_by_image = {}
    current_detection_session_dir = find_latest_session_dir() # Use the one created by setup_directories()
    
    print(f"\nGenerating detected coordinates using Haar cascades into: '{current_detection_session_dir}/'")
    for filename in image_filenames:
        image_filepath = os.path.join(IMG_DIR, filename)
        
        if not os.path.exists(image_filepath):
            print(f"Warning: Image file '{image_filepath}' not found for detection. Skipping.")
            continue

        detected_output = detect_eyes_and_save(image_filepath, face_cascade, eye_cascade, current_detection_session_dir)
        if detected_output:
            detected_coordinates_by_image[filename] = (detected_output['left_eye'], detected_output['right_eye'])

    # --- Step 3: Match Detected Faces to Ground Truth and Calculate Metrics ---
    
    # Load all ground truth and detected coordinates from files
    loaded_gt_coords = {}
    loaded_det_coords = {}

    for filename in image_filenames:
        gt_filepath = os.path.join(GT_SESSION_DIR, os.path.splitext(filename)[0] + '.gnd')
        det_filepath = os.path.join(current_detection_session_dir, os.path.splitext(filename)[0] + '.gnd')

        gt_data = load_coordinates(gt_filepath)
        if gt_data:
            loaded_gt_coords[filename] = gt_data

        det_data = load_coordinates(det_filepath)
        if det_data:
            loaded_det_coords[filename] = det_data

    if not loaded_gt_coords:
        print("\nError: No Ground Truth coordinates loaded. Cannot proceed with matching and metrics.")
        exit()
    if not loaded_det_coords:
        print("\nWarning: No Detected coordinates loaded. Metrics will likely show all False Negatives.")
    
    # Perform matching
    matching_threshold = 20 # Pixels. Adjust this value based on your image resolution and desired strictness
    matched_pairs, false_positives, false_negatives = \
        match_faces_to_ground_truth(loaded_gt_coords, loaded_det_coords, threshold=matching_threshold)

    # Calculate and print metrics
    calculate_and_print_metrics(matched_pairs, false_positives, false_negatives)

    print(f"\nFinal .gnd files for detected coordinates are in: '{current_detection_session_dir}/'")
    print(f"Ground Truth .gnd files are in: '{GT_SESSION_DIR}/'")
    print("Program execution completed.")