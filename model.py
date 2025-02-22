import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # You can also try LogisticRegression.
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model_from_csv(csv_file):
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Expected CSV columns:
    # "image", "background_R", "background_G", "background_B", 
    # "flake_R", "flake_G", "flake_B", "label"
    # We'll use the 6 color columns as features.
    feature_columns = ["background_R", "background_G", "background_B",
                       "flake_R", "flake_G", "flake_B"]
    X = df[feature_columns].values
    y = df["label"].values
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the classifier.
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the classifier on the test set.
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf

if __name__ == "__main__":
    # Replace with your CSV file path.
    csv_file_path = "combined_data.csv"
    model = train_model_from_csv(csv_file_path)
    
    # Save the trained model to a file
    model_filename = "flake_classifier.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
    df = pd.read_csv("combined_data.csv")
    print(df["label"].value_counts())
