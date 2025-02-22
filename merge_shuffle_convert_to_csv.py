import json
import csv
import random

def combine_jsons_to_csv(json_file1, json_file2, csv_output):
    """
    Combines two JSON files containing labeled data, shuffles the combined dataset,
    and writes the result to a CSV file.

    Expected JSON structure:
        {
            "image_filename1": [
                [background_color, flake_color, label],
                ...
            ],
            "image_filename2": [
                [background_color, flake_color, label],
                ...
            ],
            ...
        }
    where:
        - background_color and flake_color are lists of three integers each.
        - label is an integer (e.g., 1 or 0).

    The CSV will have columns:
        image, background_R, background_G, background_B, flake_R, flake_G, flake_B, label
    """
    data_list = []

    for json_file in [json_file1, json_file2]:
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Process each image key in the JSON
        for image_filename, samples in data.items():
            for sample in samples:
                # We expect sample to be in the form [background, flake, label]
                if len(sample) != 3:
                    continue  # Skip malformed samples
                background, flake, label = sample
                # Build a row: image filename, background channels, flake channels, label
                row = [image_filename] + background + flake + [label]
                data_list.append(row)

    # Shuffle the combined dataset to avoid any ordering bias.
    random.shuffle(data_list)

    # Write the combined data to a CSV file.
    with open(csv_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row.
        writer.writerow(["image", "background_R", "background_G", "background_B",
                         "flake_R", "flake_G", "flake_B", "label"])
        # Write each row.
        writer.writerows(data_list)

    print(f"Combined and shuffled CSV data has been saved to {csv_output}")

if __name__ == "__main__":
    # Replace these with your actual JSON file paths.
    json_file_with_label1 = "data_true_labeled.json"  # e.g., flakes with few layers (label 1)
    json_file_with_label0 = "data_false_labeled.json"  # e.g., flakes with more than few layers (label 0)
    csv_output_file = "combined_data.csv"

    combine_jsons_to_csv(json_file_with_label1, json_file_with_label0, csv_output_file)
