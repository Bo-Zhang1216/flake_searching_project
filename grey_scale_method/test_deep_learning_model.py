import cv2
import os
import glob
import numpy as np
import sys
import tensorflow as tf

def compute_features(image, x, y):
    """
    Converts the original BGR image to grayscale, normalizes it, and computes the feature vector:
        [I_background, I_flake, Delta_I, Ratio]
    where:
        I_background = mode of all pixel intensities in the normalized grayscale image
        I_flake = intensity at the clicked point (x,y)
        Delta_I = I_flake - I_background
        Ratio = I_flake / I_background (or 0 if I_background == 0)
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
    Returns a sorted list of image file paths from the specified folder.
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
    image_paths.sort()
    return image_paths

def load_and_display_image(image_path, window_name):
    """
    Loads the image, overlays the filename, and displays it.
    Returns both the original image and a copy for annotation.
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

def main(folder, model_filename="complex_model.h5"):
    # Load the Keras model.
    model = tf.keras.models.load_model(model_filename)
    print(f"Model loaded from {model_filename}")

    # Load all image paths from the folder.
    image_paths = load_images_from_folder(folder)
    if not image_paths:
        print("No images found in the folder.")
        return

    current_index = 0
    window_name = "Test Model - Click to classify (a: prev, d: next, Esc: quit)"
    cv2.namedWindow(window_name)

    # Load the first image.
    original, display_img = load_and_display_image(image_paths[current_index], window_name)
    if original is None:
        return

    def mouse_callback(event, x, y, flags, param):
        nonlocal original, display_img, model
        if event == cv2.EVENT_LBUTTONDOWN:
            # Compute feature vector for the clicked point.
            features = compute_features(original, x, y)
            features_np = np.array(features, dtype=np.float32).reshape(1, -1)
            # Get model prediction.
            prob = model.predict(features_np, verbose=0)[0][0]
            pred = 1 if prob >= 0.5 else 0
            result_text = "Flake" if pred == 1 else "Not Flake"
            # Draw a circle and overlay the result (with probability) on the image.
            cv2.circle(display_img, (x, y), 5, (0,255,0), -1)
            cv2.putText(display_img, f"{result_text} ({prob:.2f})", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow(window_name, display_img)

    cv2.setMouseCallback(window_name, mouse_callback)

    print("Instructions:")
    print("  - Click on the image to classify the point.")
    print("  - Press 'd' for the next image, 'a' for the previous image.")
    print("  - Press Esc to exit.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # Esc key
            break
        elif key == ord('d'):
            if current_index < len(image_paths) - 1:
                current_index += 1
                original, display_img = load_and_display_image(image_paths[current_index], window_name)
            else:
                print("This is the last image.")
        elif key == ord('a'):
            if current_index > 0:
                current_index -= 1
                original, display_img = load_and_display_image(image_paths[current_index], window_name)
            else:
                print("This is the first image.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_folder="/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/deep_learning_data"
    model_file = "/Users/massimozhang/Desktop/coding/Ma Lab/flake_searching_project/grey_scale_method/complex_model.h5"
    main(image_folder, model_filename=model_file)
