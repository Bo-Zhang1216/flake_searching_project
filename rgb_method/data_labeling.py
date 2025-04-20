import json

def add_label_to_data(json_path, label, output_path="data_labeled.json"):
    """
    Loads the JSON file created by the image annotation code, 
    appends the specified label to each saved set, and writes
    the result to a new JSON file.
    
    Parameters:
        json_path (str): Path to the existing JSON file.
        label (int): The label to add (e.g., 1 for few layers, 0 for not).
        output_path (str): Path for the output labeled JSON file.
    """
    # Load existing data
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Loop over each image and each set, appending the label.
    for img_file in data:
        updated_sets = []
        for flake_set in data[img_file]:
            # Check if this set already has a label (avoid duplicate labeling)
            if len(flake_set) == 2:
                # Append the label, making the set [background, flake, label]
                updated_sets.append(flake_set + [label])
            else:
                # If already labeled, you might decide to update or skip
                updated_sets.append(flake_set)
        data[img_file] = updated_sets
    
    # Write the updated data to a new JSON file.
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Labeled data has been saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Set the path to your data file and desired label.
    json_file_path = "/Users/massimozhang/Desktop/coding/Ma Lab/flake_searching_project/rgb_method/data_false.json"  # Path to your original JSON file
    label = 0  # For example, 1 for few layers, 0 for not
    add_label_to_data(json_file_path, label)
