import cv2
import os
import datetime
import math # For calculating Euclidean distance

# --- Directory Setup ---
IMG_DIR = "img"
GND_BASE_DIR = "gnd_files"
HAARCASCADE_DIR = "." # Assuming haarcascade XMLs are in the same directory as the script
GND_SESSION_DIR = None # This will be set dynamically

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "haarcascade_eye.xml"

def setup_directories():
    """
    Creates the necessary directories if they don't exist, including a timestamped
    directory for .gnd files.
    """
    global GND_SESSION_DIR

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(GND_BASE_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    GND_SESSION_DIR = os.path.join(GND_BASE_DIR, f"session_{timestamp}")
    os.makedirs(GND_SESSION_DIR, exist_ok=True)
    print(f"Ensured '{IMG_DIR}/' exists.")
    print(f"Created session directory for coordinates: '{GND_SESSION_DIR}/'")

# --- Core Detection Logic ---
def detect_eyes_and_save(image_filepath, face_cascade, eye_cascade):
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
        save_coordinates_to_file(img_filename, best_left_eye, best_right_eye)
        return {
            'image': img_filename,
            'left_eye': best_left_eye,
            'right_eye': best_right_eye
        }
    else:
        print(f"  Could not reliably detect a pair of left/right eyes in {img_filename}.")
        return None

def save_coordinates_to_file(filename, left_eye, right_eye):
    """
    Saves the provided eye coordinates to a .gnd file in the timestamped GND_SESSION_DIR.
    """
    gnd_filename = os.path.splitext(filename)[0] + '.gnd'
    gnd_filepath = os.path.join(GND_SESSION_DIR, gnd_filename)
    
    with open(gnd_filepath, 'w') as f:
        f.write(f"L: {left_eye[0]}, {left_eye[1]}\n")
        f.write(f"R: {right_eye[0]}, {right_eye[1]}\n")
    print(f"  Coordinates saved to {gnd_filepath}")

# --- Main Execution ---
if __name__ == "__main__":
    setup_directories()

    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

    if face_cascade.empty():
        print(f"Error: Face cascade XML not loaded from {FACE_CASCADE_PATH}. Check path and file integrity.")
        exit()
    if eye_cascade.empty():
        print(f"Error: Eye cascade XML not loaded from {EYE_CASCADE_PATH}. Check path and file integrity.")
        exit()

    # List of image filenames (expected to be in the IMG_DIR)
    # These are the files you uploaded, assuming they are in the 'img' folder.
    image_filenames_to_process = [
        "train1.jpg", "train2.jpg", "train3.jpg", "train4.jpg", "train5.jpg",
        "sample.jpg", "test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg"
    ]

    all_processed_coordinates = []

    print("\nStarting automated eye detection process.")
    print(f"Images will be read from '{IMG_DIR}/'.")
    print(f"Detected coordinates will be saved in a timestamped folder within '{GND_BASE_DIR}/'.")
    print("This process will run without displaying images.")

    for filename in image_filenames_to_process:
        image_filepath = os.path.join(IMG_DIR, filename)
        
        # Check if the image file exists
        if not os.path.exists(image_filepath):
            print(f"Warning: Image file '{image_filepath}' not found. Skipping.")
            continue

        print(f"Processing image: {filename}")
        # Run detection and save coordinates
        detected_coords = detect_eyes_and_save(image_filepath, face_cascade, eye_cascade)
        if detected_coords:
            all_processed_coordinates.append(detected_coords)

    print("\n--- Eye Detection Summary ---")
    if all_processed_coordinates:
        for entry in all_processed_coordinates:
            print(f"Image: {entry['image']}")
            print(f"  Left Eye: L: {entry['left_eye'][0]}, {entry['left_eye'][1]}")
            print(f"  Right Eye: R: {entry['right_eye'][0]}, {entry['right_eye'][1]}")
    else:
        print("No eyes were successfully detected and saved for any images.")
    print(f"\nAll .gnd files saved in: '{GND_SESSION_DIR}/'")
    print("Automated eye detection process completed.")