import os
import sys
import cv2
import numpy as np
from tensorflow import keras

def compute_background_color(img_rgb):
    """
    Compute the background color by taking the mode (most common value)
    for each RGB channel.
    """
    background = []
    for i in range(3):
        channel = img_rgb[:, :, i].flatten()
        mode_val = int(np.bincount(channel).argmax())
        background.append(mode_val)
    return background

# Load the saved model once
model = keras.models.load_model('my_model_with_background.h5')
print("Loaded model from 'my_model.h5'")

# Global variables to store current image data for the mouse callback
current_img_rgb = None       # The image in RGB (for pixel access)
current_img_display = None   # A BGR copy for display
current_background = None    # Computed background color
current_filename = None      # Current image file name

def mouse_callback(event, x, y, flags, param):
    """
    On left mouse click, get the pixel's color (flake color), combine with the
    background color, predict using the loaded model, and overlay the result on the image.
    """
    global current_img_rgb, current_img_display, current_background
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get flake color at (x, y) from the RGB image
        flake_color = current_img_rgb[y, x].tolist()
        print(f"Clicked at ({x}, {y})")
        print("Flake color (RGB):", flake_color)
        print("Background color (RGB):", current_background)
        # Create feature vector: background + flake, then normalize
        input_features = np.array(current_background + flake_color, dtype=np.float32).reshape(1, -1) / 255.0
        prediction = model.predict(input_features)
        result_text = "True" if prediction[0][0] > 0.5 else "False"
        print("Prediction:", result_text)
        # Draw a circle at the click and overlay the result text on the display image
        cv2.circle(current_img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(current_img_display, result_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        cv2.imshow('Image', current_img_display)

def main():
    global current_img_rgb, current_img_display, current_background, current_filename
    
    folder_path = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/deep_learning_data"

    # Get list of image files (jpg, jpeg, png, bmp)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(valid_extensions)]
    if not image_files:
        print("No image files found in the specified folder.")
        sys.exit(1)
    image_files.sort()  # Sort for consistent order
    index = 0

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    while True:
        current_filename = image_files[index]
        img_bgr = cv2.imread(current_filename)
        if img_bgr is None:
            print("Failed to load image:", current_filename)
            # Skip to next image
            index = (index + 1) % len(image_files)
            continue

        # Create a copy for display and convert to RGB for processing
        current_img_display = img_bgr.copy()
        current_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        current_background = compute_background_color(current_img_rgb)
        print(f"Displaying {current_filename}")
        print("Computed background color (RGB):", current_background)
        cv2.imshow('Image', current_img_display)

        # Wait for a key press:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('a'):  # previous image
            index = (index - 1) % len(image_files)
        elif key == ord('d'):  # next image
            index = (index + 1) % len(image_files)
        # Otherwise, continue showing the current image until a valid key is pressed

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
