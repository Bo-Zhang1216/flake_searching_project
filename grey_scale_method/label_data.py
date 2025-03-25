import json
import os

def apply_label_to_data(json_file_path, label, output_file_path=None):
    """
    Loads a JSON file with feature vectors, appends the specified label to each vector,
    and saves the updated data to a new JSON file (or overwrites the original if no output file is given).

    Parameters:
        json_file_path (str): Path to the JSON file with the feature vectors.
        label (any): The label to apply (e.g. "class1" or 1).
        output_file_path (str, optional): Where to save the updated JSON data.
                                          If not provided, the original file will be overwritten.
    """
    if output_file_path is None:
        output_file_path = json_file_path

    # Load data from the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # For each image, append the label to each feature vector if not already present.
    for image_key, feature_vectors in data.items():
        for i in range(len(feature_vectors)):
            # If the feature vector has 4 elements, add the label.
            # If it already has 5, update the last element.
            if len(feature_vectors[i]) == 4:
                feature_vectors[i].append(label)
            else:
                feature_vectors[i][-1] = label

    # Save the updated data to the output file.
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Label '{label}' applied to data in '{json_file_path}' and saved to '{output_file_path}'.")

# Example usage:
if __name__ == "__main__":
    # Set the path to your JSON file and the label you want to apply.
    json_file = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/grey_scale_method/data_false_grey.json"      # Update with your file path if needed.
    label_to_apply = "0"       # Replace with the label you wish to assign.
    output_file = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/grey_scale_method/data_false_grey_labeled.json"  # Optional: specify a new output file name.

    apply_label_to_data(json_file, label_to_apply, output_file)
