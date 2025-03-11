import os
print("1")
from PIL import Image
print("3")

from tqdm import tqdm
print("2")

def convert_tif_to_jpg(folder_path):
    # List all .tif files (case-insensitive) in the folder.
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    
    total_files = len(tif_files)
    if total_files == 0:
        print("No .tif files found in the folder.")
        return

    print(f"Found {total_files} .tif files to convert.")
    
    # Process files with a progress bar and detailed output.
    for idx, tif_file in enumerate(tqdm(tif_files, desc="Converting files", unit="file"), start=1):
        tif_path = os.path.join(folder_path, tif_file)
        try:
            with Image.open(tif_path) as img:
                # Convert to RGB (JPEG doesn't support alpha channels)
                rgb_img = img.convert('RGB')
                # Create the output file name with .jpg extension
                jpg_file = os.path.splitext(tif_file)[0] + '.jpg'
                jpg_path = os.path.join(folder_path, jpg_file)
                rgb_img.save(jpg_path, 'JPEG')
            # print(f"[{idx}/{total_files}] Converted: {tif_file} -> {jpg_file}")
        except Exception as e:
            print(f"[{idx}/{total_files}] Error converting {tif_file}: {e}")

if __name__ == '__main__':
    folder_path ="/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/deep_learning_data"
    if os.path.isdir(folder_path):
        convert_tif_to_jpg(folder_path)
        print("Conversion complete!")
    else:
        print("The provided path is not a valid directory.")
