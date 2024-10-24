# src/decision_tree.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np


def run_decision_tree_with_gridsearch():
    """Load the dataset, perform GridSearchCV to optimize hyperparameters, and run a Decision Tree with test set validation."""

    # Load the dataset
    data = pd.read_csv(
        "C:/Users/BossJore/PycharmProjects/python_SQL/youth_addiction/data/processed/youth_smoking_drug_data_final_preprocessed.csv")

    # Define the feature set and target variable
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

    # Target variable: Smoking prevalence
    y = np.where(data['scaler__Smoking_Prevalence'] > 0.5, 1, 0)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Decision Tree Classifier model
    model = DecisionTreeClassifier(random_state=42)

    # Set up the parameter grid for GridSearch
    param_grid = {
        'max_depth': [3, 5, 10],  # Constrain the depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
        'criterion': ['gini', 'entropy']  # The function to measure the quality of a split
    }

    # Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    # Perform GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best parameters and best score
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Cross-Validated Accuracy from GridSearch: ", grid_search.best_score_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Fit the best model and make predictions on the test set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Output model performance metrics on the test set
    print("Decision Tree Metrics on Test Set")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='binary'))
    print("Recall:", recall_score(y_test, y_pred, average='binary'))

    # Output Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print("\nFeature Importance (Top 10):")
    print(feature_importance.head(10))


if __name__ == "__main__":
    run_decision_tree_with_gridsearch()
