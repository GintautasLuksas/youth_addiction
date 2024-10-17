# src/naive_bayes.py
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def run_naive_bayes():
    data = pd.read_csv("../data/processed/youth_smoking_drug_data_final_preprocessed.csv")
    X = data.drop(columns=['Target_Column'])
    y = data['Target_Column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Naive Bayes Metrics")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='binary'))
    print("Recall:", recall_score(y_test, y_pred, average='binary'))
