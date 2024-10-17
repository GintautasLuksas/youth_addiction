# src/random_forest.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def run_random_forest():
    """Run Random Forest Classifier with binary target and evaluate using classification metrics."""
    data = pd.read_csv("../data/processed/youth_smoking_drug_data_final_preprocessed.csv")

    data['High_Smoking_Prevalence'] = (data['scaler__Smoking_Prevalence'] > 0.5).astype(int)

    X = data.drop(columns=['scaler__Smoking_Prevalence', 'High_Smoking_Prevalence'])
    y = data['High_Smoking_Prevalence']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Random Forest Metrics")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='binary'))
    print("Recall:", recall_score(y_test, y_pred, average='binary'))


if __name__ == "__main__":
    run_random_forest()
