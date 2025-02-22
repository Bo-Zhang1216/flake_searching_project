import json
import csv

def convert_json_to_csv(json_file, csv_file):
    """
    Converts a JSON file with image color data into a CSV file suitable for the GMM model.
    
    The JSON file should have the following structure:
    
    {
        "image_filename.jpg": [
            [
                [flake_R, flake_G, flake_B],
                [background_R, background_G, background_B]
            ],
            ...
        ],
        ...
    }
    
    The resulting CSV file will have columns:
      flake_R, flake_G, flake_B, background_R, background_G, background_B
    Each row represents one pair of color measurements.
    """
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Open the CSV file for writing
    with open(csv_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        # Write header
        writer.writerow(["flake_R", "flake_G", "flake_B", "background_R", "background_G", "background_B"])
        
        # Iterate over each image and its corresponding measurements
        for image_name, measurements in data.items():
            for measurement in measurements:
                # Each measurement is a pair: [flake_color, background_color]
                flake_color, background_color = measurement
                # Combine both lists into one row
                row = flake_color + background_color
                writer.writerow(row)
    
    print(f"CSV file saved as {csv_file}")

# Example usage:
convert_json_to_csv('data_true.json', 'flake_data.csv')
