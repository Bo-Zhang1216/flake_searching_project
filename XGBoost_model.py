import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_xgboost_model(csv_file):
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Expected CSV columns:
    # "image", "background_R", "background_G", "background_B", 
    # "flake_R", "flake_G", "flake_B", "label"
    feature_columns = ["background_R", "background_G", "background_B",
                       "flake_R", "flake_G", "flake_B"]
    X = df[feature_columns].values
    y = df["label"].values
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define and train the XGBoost classifier.
    # You can adjust hyperparameters as needed.
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=200,      # You can experiment with more trees.
        max_depth=5,           # Maximum tree depth.
        learning_rate=0.05,    # Smaller learning rate for a more robust model.
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the trained model.
    model_filename = "flake_classifier_xgb.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
    
    return model

if __name__ == "__main__":
    csv_file_path = "combined_data.csv"
    train_xgboost_model(csv_file_path)
