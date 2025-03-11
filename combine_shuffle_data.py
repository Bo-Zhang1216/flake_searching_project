import json
import random
import sys

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def combine_and_shuffle(file1, file2, output_file):
    # Load both labeled datasets.
    data_true = load_json(file1)
    data_false = load_json(file2)

    combined = []

    # Process first file (e.g., labeled true) and add each data entry with its filename.
    for filename, items in data_true.items():
        for item in items:
            combined.append({
                "filename": filename,
                "data": item
            })

    # Process second file (e.g., labeled false) similarly.
    for filename, items in data_false.items():
        for item in items:
            combined.append({
                "filename": filename,
                "data": item
            })

    # Shuffle the combined list randomly.
    random.shuffle(combined)

    # Write the combined and shuffled data to the output file.
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=4)

    print(f"Combined and shuffled {len(combined)} items saved to {output_file}")

if __name__ == '__main__':
    
    file1 = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/final_data.json"
    file2 = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/data_labeled_background.json"
    output_file = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/final_data_1.json"

    combine_and_shuffle(file1, file2, output_file)
