import cv2
import os
import datetime

# Define directories
IMG_DIR = "img"
GND_BASE_DIR = "gnd_files" # Base directory for all timestamped gnd folders
GND_SESSION_DIR = None # This will be set with a timestamp
HAARCASCADE_DIR = "." # Assuming haarcascade XMLs are in the same directory as the script

# Haar Cascade paths
FACE_CASCADE_PATH = os.path.join(HAARCASCADE_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(HAARCASCADE_DIR, "haarcascade_eye.xml")

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

def detect_eyes_and_save(image_filepath, face_cascade, eye_cascade):
    """
    Detects faces and then eyes within faces, estimates eye centers,
    saves to a .gnd file, and returns the detected coordinates.
    """
    img_filename = os.path.basename(image_filepath)
    img = cv2.imread(image_filepath)
    if img is None:
        print(f"Error: Could not load image {image_filepath}. Skipping.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    detected_eyes = []
    
    if len(faces) == 0:
        print(f"No faces detected in {img_filename}. Skipping eye detection for this image.")
        return None

    # For simplicity, we'll assume the largest face is the primary one,
    # or process all faces and take eyes from the most central/largest.
    # Here, we iterate through all faces and collect all detected eyes.
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (ex, ey, ew, eh) in eyes:
            # Calculate absolute eye center coordinates
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            detected_eyes.append((eye_center_x, eye_center_y))

    if not detected_eyes:
        print(f"No eyes detected in {img_filename}. Skipping coordinate saving for this image.")
        return None

    # Try to determine left and right eyes based on x-coordinate
    # Sort detected eyes by their x-coordinate to easily get left/right
    detected_eyes.sort(key=lambda coord: coord[0])

    left_eye_coord = None
    right_eye_coord = None

    if len(detected_eyes) >= 2:
        # Assuming the leftmost is the left eye and rightmost is the right eye
        left_eye_coord = detected_eyes[0]
        right_eye_coord = detected_eyes[-1]
    elif len(detected_eyes) == 1:
        # If only one eye is detected, we can't definitively say left/right,
        # so we'll just log it as a single detection for now, or you could skip.
        # For the purpose of this script, we need both. Let's skip if only one is found.
        print(f"Only one eye detected in {img_filename}. Cannot determine specific left/right. Skipping.")
        return None

    if left_eye_coord and right_eye_coord:
        save_coordinates_to_file(img_filename, left_eye_coord, right_eye_coord)
        return {
            'image': img_filename,
            'left_eye': left_eye_coord,
            'right_eye': right_eye_coord
        }
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

if __name__ == "__main__":
    setup_directories()

    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

    if face_cascade.empty():
        print(f"Error: Face cascade XML not loaded from {FACE_CASCADE_PATH}. Check path.")
        exit()
    if eye_cascade.empty():
        print(f"Error: Eye cascade XML not loaded from {EYE_CASCADE_PATH}. Check path.")
        exit()

    # List of image filenames (expected to be in the IMG_DIR)
    image_filenames_to_process = [
        "train1.jpg", "train2.jpg", "train3.jpg", "train4.jpg", "train5.jpg",
        "sample.jpg", "test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg"
    ]

    all_processed_coordinates = []

    print("\nStarting automated eye detection process.")
    print(f"Images will be read from '{IMG_DIR}/'.")
    print(f"Detected coordinates will be saved in a timestamped folder within '{GND_BASE_DIR}/'.")

    for filename in image_filenames_to_process:
        image_filepath = os.path.join(IMG_DIR, filename)
        
        # Check if the image file exists
        if not os.path.exists(image_filepath):
            print(f"Warning: Image file '{image_filepath}' not found. Skipping.")
            continue

        print(f"\nProcessing image: {filename}")
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