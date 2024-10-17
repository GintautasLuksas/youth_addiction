# src/knn.py
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def run_knn():
    """Run K-Nearest Neighbors Classifier with binary target and evaluate using classification metrics."""
    data = pd.read_csv("../data/processed/youth_smoking_drug_data_final_preprocessed.csv")

    # Create binary target variable for high vs. low smoking prevalence
    data['High_Smoking_Prevalence'] = (data['scaler__Smoking_Prevalence'] > 0.5).astype(int)

    # Exclude only the target column and any irrelevant columns you don't want to use
    X = data.drop(columns=['scaler__Smoking_Prevalence', 'High_Smoking_Prevalence'])
    y = data['High_Smoking_Prevalence']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the KNN model (you can experiment with different values for n_neighbors)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn_model.predict(X_test)

    # Evaluate the model using classification metrics
    print("K-Nearest Neighbors Metrics")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='binary'))
    print("Recall:", recall_score(y_test, y_pred, average='binary'))


if __name__ == "__main__":
    run_knn()
