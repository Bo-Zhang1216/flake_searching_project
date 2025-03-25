import json
import random
import sys

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def combine_and_shuffle(file_paths, output_file):
    combined = []
    
    # Process each file.
    for file_path in file_paths:
        data = load_json(file_path)
        # Assume each file is a dictionary: filename -> list of items.
        for filename, items in data.items():
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
    # Define the three files you want to combine.
    file1 = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/data_labeled_true.json"
    file2 = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/data_labeled_false_1.json"
    file3 = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/data_labeled_false.json"  # Replace with your third file path

    output_file = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/final_data_1.json"

    combine_and_shuffle([file1, file2, file3], output_file)
