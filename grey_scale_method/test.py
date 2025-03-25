import cv2
import os
import glob
import pickle
import numpy as np
import sys

def compute_features(image, x, y):
    """
    Given a BGR image and a click coordinate (x,y), convert to grayscale,
    normalize intensities, compute the mode (I_background), then extract:
       I_flake: intensity at (x,y)
       Delta_I: I_flake - I_background
       Ratio: I_flake / I_background (or 0 if I_background is 0)
    Returns a feature vector of four values.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    flat = processed_gray.flatten()
    I_background = int(np.bincount(flat).argmax())
    I_flake = int(processed_gray[y, x])
    Delta_I = I_flake - I_background
    Ratio = I_flake / I_background if I_background != 0 else 0
    return [I_background, I_flake, Delta_I, Ratio]

def load_images_from_folder(folder):
    """
    Returns a sorted list of image paths from the folder.
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
    image_paths.sort()
    return image_paths

def load_and_display_image(image_path, window_name):
    """
    Loads the image, creates a copy for annotation, and displays it.
    Overlays the filename on the image.
    Returns the original image and the annotated copy.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image:", image_path)
        return None, None
    display_img = image.copy()
    filename = os.path.basename(image_path)
    cv2.putText(display_img, filename, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow(window_name, display_img)
    return image, display_img

def main(folder, model_filename="model.pkl"):
    # Load the model from the given file (assumed to be in the same folder as the script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")

    # Load image paths
    image_paths = load_images_from_folder(folder)
    if not image_paths:
        print("No images found in the specified folder.")
        return

    current_index = 0
    window_name = "Test Model - Click to classify (a: prev, d: next, Esc: quit)"
    cv2.namedWindow(window_name)
    
    # Load the first image
    original, display_img = load_and_display_image(image_paths[current_index], window_name)
    if original is None:
        return

    def mouse_callback(event, x, y, flags, param):
        nonlocal original, display_img, model
        if event == cv2.EVENT_LBUTTONDOWN:
            # Compute feature vector from the processed version of the image.
            features = compute_features(original, x, y)
            features_np = np.array(features).reshape(1, -1)
            pred = model.predict(features_np)[0]
            result_text = "Flake" if pred == 1 else "Not Flake"
            # Draw a circle and the prediction text on the displayed image.
            cv2.circle(display_img, (x, y), 5, (0,255,0), -1)
            cv2.putText(display_img, result_text, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow(window_name, display_img)
    
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Instructions:")
    print("  - Click on the image to classify the point.")
    print("  - Press 'd' to go to the next image, 'a' to go to the previous image.")
    print("  - Press Esc to exit.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # Esc key
            break
        elif key == ord('d'):
            # Next image
            if current_index < len(image_paths) - 1:
                current_index += 1
                original, display_img = load_and_display_image(image_paths[current_index], window_name)
            else:
                print("This is the last image.")
        elif key == ord('a'):
            # Previous image
            if current_index > 0:
                current_index -= 1
                original, display_img = load_and_display_image(image_paths[current_index], window_name)
            else:
                print("This is the first image.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder = "/Users/massimozhang/Downloads/2DMatGMM-main/Datasets/GMMDetectorDatasets/Graphene/train_images"
    folder="/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/deep_learning_data"
    model_file = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/grey_scale_method/decision_tree_model.pkl"
    main(folder,model_filename=model_file)
