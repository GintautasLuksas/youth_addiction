import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def run_knn_with_gridsearch():
    """
    Load the dataset, split it into training and test sets, and run a K-Nearest Neighbors (KNN) model
    with GridSearchCV to optimize hyperparameters.

    The function uses KNN to classify smoking prevalence based on several features.
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

    knn_model = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Cross-Validated Accuracy from GridSearch: ", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("KNN Metrics on Test Set")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='binary'))
    print("Recall:", recall_score(y_test, y_pred, average='binary'))

if __name__ == "__main__":
    run_knn_with_gridsearch()
