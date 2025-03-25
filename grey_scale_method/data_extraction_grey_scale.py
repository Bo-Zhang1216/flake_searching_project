import cv2
import os
import glob
import json
import numpy as np

def main(folder=None):
    # Ask for the folder containing images if not provided.
    if not folder:
        folder = input("Enter the path to the folder containing images: ").strip()

    # Gather image paths (feel free to add more extensions if needed)
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
    image_paths.sort()
    if not image_paths:
        print("No images found in the provided folder.")
        return

    # Dictionary to store saved feature vectors.
    # Key: image filename; Value: list of feature vectors for each set.
    saved_data = {}

    current_image_index = 0
    current_clicks = []  # stores (x, y) for the current unsaved set (flake points)

    # Global images
    original_image = None    # original loaded image (BGR)
    processed_gray = None    # processed grayscale image (max contrast)
    processed_color = None   # processed image in BGR (converted from grayscale for display)
    display_image = None     # combined image for display (original on left, processed on right)

    def draw_overlay(img, filename):
        """Draw overlay text on an image:
           - Filename in blue at the bottom.
           - Saved set count in red at the top if available.
        """
        overlay = img.copy()
        cv2.putText(overlay, filename, (10, overlay.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if filename in saved_data and saved_data[filename]:
            text = f"Saved sets: {len(saved_data[filename])}"
            cv2.putText(overlay, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return overlay

    def load_image(index):
        """Loads an image by index, computes the processed version, resets current clicks,
           and refreshes the combined display.
        """
        nonlocal original_image, processed_gray, processed_color, display_image, current_clicks
        image_path = image_paths[index]
        filename = os.path.basename(image_path)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print("Could not load image:", image_path)
            return

        # Process the image:
        # Convert to grayscale and apply normalization to stretch intensities (autocontrast effect).
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        processed_color = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)

        current_clicks = []  # reset unsaved selection
        refresh_display()
        cv2.imshow("Image", display_image)

    def refresh_display():
        """Redraws the combined display image with overlays and any current unsaved points.
           The left half shows the original image and the right half shows the processed image.
        """
        nonlocal display_image
        filename = os.path.basename(image_paths[current_image_index])

        left_img = original_image.copy()
        left_img = draw_overlay(left_img, filename)
        right_img = processed_color.copy()
        right_img = draw_overlay(right_img, filename)

        # Draw any current (unsaved) clicks as green circles on both images.
        for (x, y) in current_clicks:
            cv2.circle(left_img, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(right_img, (x, y), 5, (0, 255, 0), -1)

        # Combine the images side by side.
        display_image = np.hstack([left_img, right_img])
        cv2.imshow("Image", display_image)

    # Create a window and set up the mouse callback.
    cv2.namedWindow("Image")
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_clicks, display_image
        if original_image is None:
            return

        img_width = original_image.shape[1]
        if event == cv2.EVENT_LBUTTONDOWN:
            # If click is on the left half, coordinates are as is.
            # If click is on the right half, subtract the width offset.
            if x < img_width:
                point = (x, y)
            else:
                point = (x - img_width, y)
            current_clicks.append(point)
            refresh_display()
    cv2.setMouseCallback("Image", mouse_callback)

    # Load the first image.
    load_image(current_image_index)

    # Print instructions.
    print("\nInstructions:")
    print("  Left-click on either half of the displayed image to add a point.")
    print("     (Clicks are normalized so that the same point appears on both halves.)")
    print("  c: Clear the current unsaved point selections.")
    print("  s: Save the current set (requires at least 1 point).")
    print("       Feature Extraction (grayscale):")
    print("         I_background: Mode of processed image intensities")
    print("         I_flake: Intensity at the clicked point (from processed image)")
    print("         Delta_I: I_flake - I_background")
    print("         Ratio: I_flake / I_background (if I_background != 0)")
    print("         Combined feature vector: [I_background, I_flake, Delta_I, Ratio]")
    print("  d: Next image.")
    print("  a: Previous image.")
    print("  Esc: Quit and save data.json.\n")

    while True:
        cv2.imshow("Image", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Esc key pressed: exit.
            break
        elif key == ord('c'):
            current_clicks = []
            refresh_display()
            print("Cleared current point selections for image:", os.path.basename(image_paths[current_image_index]))
        elif key == ord('s'):
            if current_clicks:
                # Compute the background intensity as the mode of all pixels in the processed image.
                flat = processed_gray.flatten()
                I_background = int(np.bincount(flat).argmax())
                filename = os.path.basename(image_paths[current_image_index])
                if filename not in saved_data:
                    saved_data[filename] = []
                # Save a feature vector for each selected point.
                for (x, y) in current_clicks:
                    I_flake = int(processed_gray[y, x])
                    Delta_I = I_flake - I_background
                    Ratio = I_flake / I_background if I_background != 0 else 0
                    feature_vector = [I_background, I_flake, Delta_I, Ratio]
                    saved_data[filename].append(feature_vector)
                    print(f"Saved point for {filename}: {feature_vector}")
                current_clicks = []
                refresh_display()
            else:
                print("Error: Please select at least 1 point before saving.")
        elif key == ord('d'):
            if current_image_index < len(image_paths) - 1:
                current_image_index += 1
                load_image(current_image_index)
                print("Moved to next image:", os.path.basename(image_paths[current_image_index]))
            else:
                print("This is the last image.")
        elif key == ord('a'):
            if current_image_index > 0:
                current_image_index -= 1
                load_image(current_image_index)
                print("Moved to previous image:", os.path.basename(image_paths[current_image_index]))
            else:
                print("This is the first image.")

    cv2.destroyAllWindows()
    # Save the data.json file in the same directory as this script.
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_directory, "data.json")
    with open(data_path, "w") as f:
        json.dump(saved_data, f, indent=4)
    print(f"\nData saved to {data_path}")

if __name__ == "__main__":
    pathname = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/deep_learning_data"
    main(folder=pathname)
