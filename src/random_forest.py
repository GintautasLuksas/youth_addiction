import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def run_random_forest_on_preprocessed_data():
    """
    Load the preprocessed dataset, split it into training and test sets, and run a Random Forest model
    with GridSearchCV to optimize hyperparameters.

    This function uses Random Forest to classify smoking prevalence based on several preprocessed features.
    It outputs model performance metrics such as accuracy, precision, and recall on the test set.
    """

    # Load the preprocessed dataset
    data = pd.read_csv("C:/Users/BossJore/PycharmProjects/python_SQL/youth_addiction/data/processed/youth_smoking_drug_data_final_preprocessed.csv")

    # Define the feature columns (all except 'scaler__Smoking_Prevalence')
    X = data.drop(columns=['scaler__Smoking_Prevalence'])

    # Set the target variable (binary classification based on 'scaler__Smoking_Prevalence')
    y = np.where(data['scaler__Smoking_Prevalence'] > 0.5, 1, 0)  # Classifying >0.5 as a smoker

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Random Forest model
    model = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [5, 10, 20],  # Number of trees in the forest
        'max_depth': [3, 5, 15],      # Maximum depth of the tree
        'min_samples_split': [2],     # Minimum number of samples required to split a node
        'min_samples_leaf': [1],      # Minimum number of samples required at each leaf node
        'bootstrap': [True]           # Bootstrap samples when building trees
    }

    # Apply GridSearchCV for hyperparameter tuning with 5-fold cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

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
    print("Random Forest Metrics on Test Set")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='binary'))
    print("Recall:", recall_score(y_test, y_pred, average='binary'))


if __name__ == "__main__":
    run_random_forest_on_preprocessed_data()
