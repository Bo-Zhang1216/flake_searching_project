import cv2
import numpy as np
import joblib
import os
import glob

# Load the trained model
model = joblib.load("flake_classifier_xgb.joblib")
print("Model loaded successfully.")

# Global variables to store the clicked point, current image, image list, and current index
clicked_point = None
image = None
image_paths = []
current_index = 0

def compute_background_mode(img):
    """
    Compute the mode for each RGB channel.
    OpenCV loads images in BGR, so convert to RGB first.
    Returns a list: [mode_red, mode_green, mode_blue]
    """
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    red_mode = int(np.bincount(rgb_image[:, :, 0].flatten()).argmax())
    green_mode = int(np.bincount(rgb_image[:, :, 1].flatten()).argmax())
    blue_mode = int(np.bincount(rgb_image[:, :, 2].flatten()).argmax())
    return [red_mode, green_mode, blue_mode]

def mouse_callback(event, x, y, flags, param):
    global clicked_point, image
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        # Mark the clicked point on the image.
        display_img = image.copy()
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Test Image", display_img)
        
        # Get the flake color from the clicked point (convert BGR to RGB)
        b, g, r = image[y, x]
        flake_color = [int(r), int(g), int(b)]
        
        # Compute the background color from the entire image.
        background_color = compute_background_mode(image)
        
        # Build the feature vector: background (3 values) + flake color (3 values)
        feature_vector = background_color + flake_color
        feature_vector = [feature_vector]  # Make it 2D for prediction
        
        # Predict using the loaded model.
        prediction = model.predict(feature_vector)[0]
        label_str = "Few Layers" if prediction == 1 else "More than Few Layers"
        print(f"Prediction: {label_str}")
        
        # Display the prediction on the image.
        cv2.putText(display_img, f"Prediction: {label_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Test Image", display_img)

def main(folder_path=None):
    global image, image_paths, current_index
    if folder_path is None:
        folder_path = input("Enter the path to the folder containing test images: ").strip()
    
    # Collect image paths from the folder using common image extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    image_paths.sort()
    
    if not image_paths:
        print("No images found in folder:", folder_path)
        return
    
    current_index = 0
    image = cv2.imread(image_paths[current_index])
    if image is None:
        print("Error: Could not load image:", image_paths[current_index])
        return
    
    # Set up window and mouse callback
    cv2.namedWindow("Test Image")
    cv2.setMouseCallback("Test Image", mouse_callback)
    
    print("Navigation instructions:")
    print("  Press 'd' for next image")
    print("  Press 'a' for previous image")
    print("  Click on the image to test the model")
    print("  Press 'Esc' to exit")
    
    while True:
        cv2.imshow("Test Image", image)
        key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
        if key == 27:  # Esc key to exit
            break
        elif key == ord('d'):  # Next image
            if current_index < len(image_paths) - 1:
                current_index += 1
                image = cv2.imread(image_paths[current_index])
                print("Showing image:", image_paths[current_index])
            else:
                print("This is the last image.")
        elif key == ord('a'):  # Previous image
            if current_index > 0:
                current_index -= 1
                image = cv2.imread(image_paths[current_index])
                print("Showing image:", image_paths[current_index])
            else:
                print("This is the first image.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(folder_path="/Users/massimozhang/Downloads/2DMatGMM-main/Datasets/GMMDetectorDatasets/Graphene/train_images_2")
