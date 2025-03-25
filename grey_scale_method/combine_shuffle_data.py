import json
import random

def combine_and_shuffle_data(file1, file2, output_file_path=None):
    """
    Loads two labeled JSON data files, combines all feature vectors, shuffles the combined list,
    and optionally writes the result to a new JSON file.

    Each JSON file is expected to be a dictionary where keys are identifiers (e.g. image filenames)
    and values are lists of feature vectors (each a list with 5 elements, where the last element is the label).

    Parameters:
        file1 (str): Path to the first JSON data file.
        file2 (str): Path to the second JSON data file.
        output_file_path (str, optional): Path to save the combined and shuffled data. 
                                          If not provided, the data is not saved to disk.

    Returns:
        list: A shuffled list of all feature vectors from both files.
    """
    # Load data from the first file
    with open(file1, 'r') as f:
        data1 = json.load(f)
    
    # Load data from the second file
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    # Combine all feature vectors from both files into one list
    combined_data = []
    for key in data1:
        combined_data.extend(data1[key])
    for key in data2:
        combined_data.extend(data2[key])
    
    # Shuffle the combined list randomly
    random.shuffle(combined_data)
    
    # Save the combined data to a file if an output file path is provided
    if output_file_path:
        with open(output_file_path, 'w') as f:
            json.dump(combined_data, f, indent=4)
        print(f"Combined and shuffled data saved to {output_file_path}.")
    
    return combined_data

# Example usage:
if __name__ == "__main__":
    # Replace these file paths with the paths to your labeled data files.
    file1 = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/grey_scale_method/data_false_grey_labeled.json"
    file2 = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/grey_scale_method/data_true_grey_labeled.json"
    output_file = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/grey_scale_method/combined_shuffled_data.json"
    
    combined = combine_and_shuffle_data(file1, file2, output_file)
    print("Number of combined feature vectors:", len(combined))
