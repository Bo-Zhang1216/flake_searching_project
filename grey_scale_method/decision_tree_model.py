import json
import os
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(json_file_path):
    """
    Loads the JSON file containing the combined and shuffled data.
    Each record is expected to be in the form:
         [I_background, I_flake, Delta_I, Ratio, label]
    Returns:
         X: A numpy array of shape (n_samples, 4) containing the features.
         y: A numpy array of shape (n_samples,) containing the labels as integers.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    X, y = [], []
    for record in data:
        features = record[:4]
        label = record[4]
        X.append(features)
        y.append(int(label))
    return np.array(X), np.array(y)

def train_decision_tree(json_file_path, model_filename="model.pkl"):
    """
    Loads data from the JSON file, trains a Decision Tree classifier (which tends to overfit),
    prints the training accuracy, and saves the trained model in the same folder as this script.
    
    Parameters:
         json_file_path (str): Path to the JSON file containing the data.
         model_filename (str): The filename for the saved model (default "model.pkl").
    """
    # Load data
    X, y = load_data(json_file_path)
    
    # Initialize and train a Decision Tree classifier.
    # We do not set a maximum depth so that the tree overfits the training data.
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    # Evaluate training accuracy.
    train_pred = clf.predict(X)
    train_acc = accuracy_score(y, train_pred)
    print(f"Training accuracy: {train_acc:.4f}")
    
    # Save the model in the same folder as the script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_filename)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")
    
    return clf

if __name__ == "__main__":
    json_file_path = "/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/grey_scale_method/combined_shuffled_data.json"  # Update this path if needed.
    train_decision_tree(json_file_path)
