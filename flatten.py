import os
import shutil
from tqdm import tqdm

def flatten_directory(root_dir):
    print(f"Scanning directory '{root_dir}' for files to flatten...")
    # Gather all files in subdirectories (not including the root folder itself)
    files_to_move = []
    for current_dir, subdirs, files in os.walk(root_dir):
        if os.path.abspath(current_dir) == os.path.abspath(root_dir):
            continue
        for file in files:
            src_path = os.path.join(current_dir, file)
            files_to_move.append(src_path)
    
    print(f"Found {len(files_to_move)} files to move.")

    # Process each file with a progress bar and print statements
    for src_path in tqdm(files_to_move, desc="Moving files", unit="file"):
        file = os.path.basename(src_path)
        dest_path = os.path.join(root_dir, file)
        
        # Handle potential name collisions by renaming files.
        if os.path.exists(dest_path):
            base, extension = os.path.splitext(file)
            counter = 1
            while os.path.exists(dest_path):
                new_name = f"{base}_{counter}{extension}"
                dest_path = os.path.join(root_dir, new_name)
                counter += 1
        
        shutil.move(src_path, dest_path)
        print(f"Moved: {src_path} -> {dest_path}")
    
    print("Finished moving files. Now cleaning up empty directories...")
    removed_dirs = 0
    # Remove empty directories after moving the files
    for current_dir, subdirs, files in os.walk(root_dir, topdown=False):
        if os.path.abspath(current_dir) == os.path.abspath(root_dir):
            continue
        if not os.listdir(current_dir):
            os.rmdir(current_dir)
            print(f"Removed empty directory: {current_dir}")
            removed_dirs += 1
            
    print(f"Cleanup complete. Removed {removed_dirs} empty directories.")

if __name__ == "__main__":
    folder_path = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/deep_learning_data"

    if os.path.isdir(folder_path):
        print("Starting the folder flattening process...")
        flatten_directory(folder_path)
        print("Folder flattening complete!")
    else:
        print("The provided path is not a valid directory.")
