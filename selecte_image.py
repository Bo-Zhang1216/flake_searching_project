import os
import cv2

def review_images(folder_path):
    # Get a sorted list of .tif files in the folder.
    tif_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
    
    if not tif_files:
        print("No .tif files found in the folder.")
        return

    print(f"Found {len(tif_files)} .tif files to review.")
    
    for file in tif_files:
        full_path = os.path.join(folder_path, file)
        print(f"\nReviewing: {full_path}")
        
        # Read the image using OpenCV.
        img = cv2.imread(full_path)
        if img is None:
            print("Error loading image, skipping:", full_path)
            continue
        
        # Display the image.
        cv2.imshow("Image Review - Press 'k' to keep, 'd' to delete, 'q' to quit", img)
        
        # Wait indefinitely for a key press.
        key = cv2.waitKey(0) & 0xFF
        
        # Check the key press.
        if key == ord('d'):
            try:
                os.remove(full_path)
                print("Deleted:", full_path)
            except Exception as e:
                print("Error deleting file:", e)
        elif key == ord('q'):
            print("Quitting review.")
            cv2.destroyAllWindows()
            break
        elif key == ord('k'):
            print("Keeping:", full_path)
        else:
            print("Unrecognized input. Keeping:", full_path)
        
        # Close the image window.
        cv2.destroyAllWindows()

if __name__ == '__main__':
    folder_path = input("/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/deep_learning_data").strip()
    if os.path.isdir(folder_path):
        review_images(folder_path)
    else:
        print("The provided path is not a valid directory.")
