import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np


def run_naive_bayes():
    """
    Load the dataset, split it into training and test sets, and run a Naive Bayes model
    to predict smoking prevalence.

    This function uses Gaussian Naive Bayes to classify smoking prevalence based on several features.
    It outputs model performance metrics such as accuracy, precision, and recall on the test set.
    """

    data = pd.read_csv(
        "C:/Users/BossJore/PycharmProjects/python_SQL/youth_addiction/data/processed/youth_smoking_drug_data_final_preprocessed.csv"
    )

    X = data[[
        'scaler__Drug_Experimentation',
        'scaler__Peer_Influence',
        'scaler__Family_Background',
        'scaler__Mental_Health',
        'scaler__Community_Support',
        'scaler__Media_Influence',
        'remainder__School_Programs',
        'remainder__Access_to_Counseling',
        'remainder__Substance_Education',
        'remainder__Year',
        'remainder__Age_Group',
        'remainder__Parental_Supervision',
        'onehot__Gender_1',
        'onehot__Gender_2',
        'onehot__Socioeconomic_Status_1',
        'onehot__Socioeconomic_Status_2'
    ]]

    y = np.where(data['scaler__Smoking_Prevalence'] > 0.5, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Naive Bayes Metrics on Test Set")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='binary'))
    print("Recall:", recall_score(y_test, y_pred, average='binary'))


if __name__ == "__main__":
    run_naive_bayes()
