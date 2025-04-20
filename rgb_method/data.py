import cv2
import os
import glob
import json
import numpy as np

def main(folder=None):
    # Ask for the folder containing images
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

    # This dictionary will store saved data.
    # Keys: image filename; Value: list of sets.
    # Each set is a list of two lists: [background_color, flake_color]
    saved_data = {}

    current_image_index = 0
    current_clicks = []   # stores (x, y) for the current unsaved set (flake point)
    original_image = None # the image loaded from disk
    display_image = None  # a copy of the image that we draw on

    def draw_overlay(img, filename):
        """Draw overlay text on the image:
           - The filename in blue at the bottom.
           - If any sets have been saved, display a red text at the top.
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
        """Loads an image by index, resets current clicks, and displays the image with overlay."""
        nonlocal original_image, display_image, current_clicks
        image_path = image_paths[index]
        filename = os.path.basename(image_path)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print("Could not load image:", image_path)
            return
        current_clicks = []  # reset unsaved selection
        display_image = original_image.copy()
        display_image = draw_overlay(display_image, filename)
        cv2.imshow("Image", display_image)

    def refresh_display():
        """Redraws the display image (with overlay and current selected points)."""
        nonlocal display_image
        filename = os.path.basename(image_paths[current_image_index])
        display_image = original_image.copy()
        display_image = draw_overlay(display_image, filename)
        # Draw any current (unsaved) clicks as green circles.
        for (x, y) in current_clicks:
            cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", display_image)

    # Create a window and set up the mouse callback.
    cv2.namedWindow("Image")
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_clicks, display_image
        if event == cv2.EVENT_LBUTTONDOWN:
            # Only allow one click (for the flake) per set.
            if len(current_clicks) < 1:
                current_clicks.append((x, y))
                cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Image", display_image)
    cv2.setMouseCallback("Image", mouse_callback)

    # Load the first image.
    load_image(current_image_index)

    # Print instructions.
    print("\nInstructions:")
    print("  Left-click: Add a point (only 1 point per set for the flake).")
    print("  c: Clear the current unsaved point selection.")
    print("  s: Save the current set (requires exactly 1 point).")
    print("       The background is computed as the mode of each channel's values.")
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
            print("Cleared current point selection for image:",
                  os.path.basename(image_paths[current_image_index]))

        elif key == ord('s'):
            # Save the current set only if exactly one point (flake) is selected.
            if len(current_clicks) == 1:
                # Get the flake color at the clicked point (convert BGR to RGB).
                x, y = current_clicks[0]
                b, g, r = original_image[y, x]
                flake_color = [int(r), int(g), int(b)]
                
                # Compute the background color by taking the mode of each individual channel.
                # Convert the image to RGB.
                rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # Compute mode for each channel separately.
                red_mode = int(np.bincount(rgb_image[:, :, 0].flatten()).argmax())
                green_mode = int(np.bincount(rgb_image[:, :, 1].flatten()).argmax())
                blue_mode = int(np.bincount(rgb_image[:, :, 2].flatten()).argmax())
                background_color = [red_mode, green_mode, blue_mode]
                
                rgb_set = [background_color, flake_color]
                filename = os.path.basename(image_paths[current_image_index])
                if filename not in saved_data:
                    saved_data[filename] = []
                saved_data[filename].append(rgb_set)
                print(f"Saved set for {filename}: Background: {background_color}, Flake: {flake_color}")
                current_clicks = []
                refresh_display()
            else:
                print("Error: Please select exactly 1 point before saving a set.")

        elif key == ord('d'):
            if current_image_index < len(image_paths) - 1:
                current_image_index += 1
                load_image(current_image_index)
                print("Moved to next image:",
                      os.path.basename(image_paths[current_image_index]))
            else:
                print("This is the last image.")

        elif key == ord('a'):
            if current_image_index > 0:
                current_image_index -= 1
                load_image(current_image_index)
                print("Moved to previous image:",
                      os.path.basename(image_paths[current_image_index]))
            else:
                print("This is the first image.")

    cv2.destroyAllWindows()

    # Save only the images for which data was saved.
    # Save the data.json file in the same directory as this script.
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_directory, "data.json")
    with open(data_path, "w") as f:
        json.dump(saved_data, f, indent=4)
    print(f"\nData saved to {data_path}")



if __name__ == "__main__":
    pathname="/Users/massimozhang/Desktop/coding/Ma Lab/flake_searching_project/TIT_data/100x"
    main(folder=pathname)

