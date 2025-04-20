import os
import cv2
import json
import numpy as np

# === CONFIGURATION ===
# Update these paths with your actual directories.
original_images_folder = "/Users/massimozhang/Desktop/coding/Ma Lab/flake_searching_project/deep_learning_data"   # Folder with your original images
masks_folder = "/Users/massimozhang/Desktop/coding/Ma Lab/flake_searching_project/generated_masks_2"       # Folder with generated masks and metadata file "masks_metadata.json"
output_folder = "/Users/massimozhang/Desktop/coding/Ma Lab/flake_searching_project/CNN_model/annotations" # Folder where annotated outputs will be saved

# Output subfolders
background_folder = os.path.join(output_folder, "background_masks")
valid_folder = os.path.join(output_folder, "valid_flakes")
nonvalid_folder = os.path.join(output_folder, "nonvalid_flakes")
os.makedirs(background_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(nonvalid_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Load metadata from mask generation (created on Colab)
metadata_path = os.path.join(masks_folder, "masks_metadata.json")
with open(metadata_path, "r") as f:
    masks_metadata = json.load(f)

# List all original images.
image_files = sorted([f for f in os.listdir(original_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
current_index = 0
annotations = {}  # Will store annotation info for each image

# Global variables for current image annotation.
current_image = None   # The original image (BGR)
current_masks = []     # List of dicts for annotation (non-background masks)
current_background_path = None  # Path of the pre-saved background mask
click_handled = False  # Flag to debounce clicks

def load_image_and_masks(image_filename):
    """Load the original image and its corresponding masks using the metadata.
       Save the largest mask (background) immediately and remove it from annotation."""
    global current_image, current_masks, current_background_path
    image_path = os.path.join(original_images_folder, image_filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_filename}")
        return None
    current_image = image.copy()
    
    # Load the masks metadata for this image.
    if image_filename not in masks_metadata:
        print(f"No masks metadata found for image {image_filename}")
        current_masks = []
        current_background_path = None
        return image
    
    mask_entries = masks_metadata[image_filename]
    current_masks = []
    for idx, entry in enumerate(mask_entries):
        mask_filename = entry["mask_filename"]
        area = entry["area"]
        mask_path = os.path.join(masks_folder, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        # Convert the mask to a binary array: values > 127 become 1.
        _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        current_masks.append({
            "mask": mask_bin,
            "area": area,
            "selected": False,
            "id": idx
        })
    # Automatically designate the largest mask as background,
    # save it immediately, and remove it from current_masks.
    if current_masks:
        largest_idx = max(range(len(current_masks)), key=lambda i: current_masks[i]["area"])
        background_entry = current_masks[largest_idx]
        base_name = os.path.splitext(image_filename)[0]
        bg_path = os.path.join(background_folder, base_name + "_background.png")
        cv2.imwrite(bg_path, background_entry["mask"] * 255)
        current_background_path = bg_path
        # Remove the background mask so it won't be displayed for annotation.
        del current_masks[largest_idx]
    else:
        current_background_path = None
    return image

def draw_overlay():
    """Create an overlay image with the masks drawn on top of the original image.
       Masks are filled with green if unselected and pink if selected, with a red border."""
    overlay = current_image.copy()
    for m in current_masks:
        # Convert binary mask to 0-255 image.
        mask = (m["mask"].astype(np.uint8)) * 255
        # Choose fill color based on selection status: pink for selected, green for unselected.
        fill_color = (255, 0, 255) if m["selected"] else (0, 255, 0)
        colored_mask = np.zeros_like(overlay, dtype=np.uint8)
        colored_mask[mask == 255] = fill_color
        # Blend the overlay with the fill.
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)
        # Find contours for the mask and draw a red border.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # Red border with thickness=2
    return overlay

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to toggle the selection of a mask when clicked.
       Only processes clicks on the left half of the combined image (overlay)."""
    global current_masks, current_image, click_handled
    img_width = current_image.shape[1]
    if event == cv2.EVENT_LBUTTONDOWN and not click_handled:
        # Only process clicks within the left half.
        if x >= img_width:
            return
        # Process the click using the coordinates at the moment of click.
        for m in reversed(current_masks):
            if m["mask"][y, x] > 0:
                m["selected"] = not m["selected"]
                break
        click_handled = True  # Mark this click as handled
    elif event == cv2.EVENT_LBUTTONUP:
        click_handled = False  # Reset when mouse button is released

def save_annotations(image_filename):
    """Save the annotations for the current image.
       The background mask is already saved; here we save valid and non-valid masks from annotation."""
    global current_masks, annotations, current_background_path
    base_name = os.path.splitext(image_filename)[0]
    save_info = {}
    
    # Add the pre-saved background mask path.
    if current_background_path is not None:
        save_info["background"] = current_background_path
    
    # Separate valid flake masks and non-valid masks from the remaining (annotated) masks.
    valid_masks = [m["mask"] for m in current_masks if m["selected"]]
    nonvalid_masks = [m["mask"] for m in current_masks if not m["selected"]]
    
    # Save valid flake masks.
    valid_paths = []
    for i, mask in enumerate(valid_masks):
        mask_path = os.path.join(valid_folder, f"{base_name}_flake_{i}.png")
        cv2.imwrite(mask_path, mask * 255)
        valid_paths.append(mask_path)
    save_info["valid"] = valid_paths
    
    # Save non-valid masks.
    nonvalid_paths = []
    for i, mask in enumerate(nonvalid_masks):
        mask_path = os.path.join(nonvalid_folder, f"{base_name}_nonvalid_{i}.png")
        cv2.imwrite(mask_path, mask * 255)
        nonvalid_paths.append(mask_path)
    save_info["nonvalid"] = nonvalid_paths
    
    # Optionally, save an overlay image for reference.
    overlay_path = os.path.join(output_folder, base_name + "_overlay.png")
    cv2.imwrite(overlay_path, draw_overlay())
    save_info["overlay"] = overlay_path
    
    annotations[image_filename] = save_info
    print(f"Annotations saved for {image_filename}")

def show_navigating_message():
    """Display an overlay with 'navigating' text and block keys until a new image is loaded."""
    nav_overlay = draw_overlay()
    cv2.putText(nav_overlay, "navigating", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    combined = np.hstack((nav_overlay, current_image))
    cv2.imshow("Annotator", combined)
    cv2.waitKey(500)

# Set up OpenCV window and mouse callback.
cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Annotator", mouse_callback)

# === MAIN INTERACTIVE LOOP ===
while True:
    if current_index < 0:
        current_index = 0
    if current_index >= len(image_files):
        print("No more images to annotate.")
        break
    
    current_file = image_files[current_index]
    load_image_and_masks(current_file)
    
    while True:
        overlay = draw_overlay()
        # Display the overlay on the left and the original image on the right.
        combined = np.hstack((overlay, current_image))
        cv2.imshow("Annotator", combined)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            save_annotations(current_file)
            current_index += 1
            break
        elif key == ord('d'):
            show_navigating_message()
            current_index += 1
            break
        elif key == ord('a'):
            show_navigating_message()
            current_index = max(0, current_index - 1)
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            with open(os.path.join(output_folder, "annotations.json"), "w") as f:
                json.dump(annotations, f, indent=2)
            print("Annotation session ended. Data saved.")
            exit(0)

cv2.destroyAllWindows()
with open(os.path.join(output_folder, "annotations.json"), "w") as f:
    json.dump(annotations, f, indent=2)
print("Annotation complete. Data saved.")
