import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def run_logistic_regression_with_gridsearch():
    """
    Load the dataset, split it into training and test sets, and run a Logistic Regression model
    with GridSearchCV to optimize hyperparameters.

    This function uses Logistic Regression to classify smoking prevalence based on several features.
    It outputs model performance metrics such as accuracy, precision, and recall on the test set.
    """

    # Load the preprocessed dataset
    data = pd.read_csv(
        "C:/Users/BossJore/PycharmProjects/python_SQL/youth_addiction/data/processed/youth_smoking_drug_data_final_preprocessed.csv"
    )

    # Define the feature columns (names must match the ones after ColumnTransformer in preprocessing)
    X = data[[
        'scaler__Drug_Experimentation',
        'scaler__Peer_Influence',
        'scaler__Family_Background',
        'scaler__Mental_Health',
        'scaler__Community_Support',
        'scaler__Media_Influence',
        'scaler__Year',
        'scaler__Parental_Supervision',
        'onehot__Gender_1',
        'onehot__Socioeconomic_Status_1',
        'remainder__School_Programs',
        'remainder__Access_to_Counseling',
        'remainder__Substance_Education'
    ]]

    # Set the target variable (binary classification based on 'Smoking_Prevalence')
    y = np.where(data['scaler__Smoking_Prevalence'] > 0.5, 1, 0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Logistic Regression model with fixed parameters
    model = LogisticRegression(random_state=42, max_iter=300)

    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        'penalty': ['l2'],  # L2 regularization
        'C': [0.1, 1, 10],  # Inverse regularization strength
        'solver': ['liblinear']  # Suitable for small datasets
    }

    # Apply GridSearchCV for hyperparameter tuning with 5-fold cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search model on the training data
    grid_search.fit(X_train, y_train)

    # Output the best hyperparameters and cross-validated accuracy
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Cross-Validated Accuracy from GridSearch: ", grid_search.best_score_)

    # Train the best model on the full training set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = best_model.predict(X_test)

    # Output evaluation metrics on the test set
    print("Logistic Regression Metrics on Test Set")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='binary'))
    print("Recall:", recall_score(y_test, y_pred, average='binary'))


if __name__ == "__main__":
    run_logistic_regression_with_gridsearch()
