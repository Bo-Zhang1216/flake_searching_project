import os
from PIL import Image, ImageOps, ImageEnhance

def process_images(input_folder, output_folder, contrast_factor=1.8):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(input_folder, filename)
            try:
                with Image.open(filepath) as img:
                    # Convert the image to grayscale using a luminosity method
                    gray_img = ImageOps.grayscale(img)
                    
                    # Enhance contrast to intensify the black & white effect
                    enhanced_img = ImageEnhance.Contrast(gray_img).enhance(contrast_factor)
                    
                    # Optionally apply autocontrast to stretch the histogram fully
                    final_img = ImageOps.autocontrast(enhanced_img)
                    
                    # Save the processed image to the output folder
                    output_path = os.path.join(output_folder, filename)
                    final_img.save(output_path)
                    print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    # Set your folder paths as variables
    input_folder = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/deep_learning_data"
    output_folder = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/black_and_white_images"
    
    # Adjust the contrast_factor if needed (1.0 leaves the image unchanged)
    process_images(input_folder, output_folder, contrast_factor=1.8)
