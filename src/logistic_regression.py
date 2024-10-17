# src/logistic_regression.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def run_logistic_regression():
    """Run Logistic Regression on the processed dataset, print evaluation metrics, and display feature importance.

    This function:
    - Loads the preprocessed dataset
    - Splits the data into training and testing sets
    - Trains a Logistic Regression model on the training data
    - Evaluates the model on the test data using accuracy, precision, and recall metrics
    - Displays the top 10 features based on their absolute coefficients
    """
    data = pd.read_csv("../data/processed/youth_smoking_drug_data_final_preprocessed.csv")

    # Assuming scaler__Smoking_Prevalence is the binary target variable, based on prevalence levels
    # Adjust threshold if necessary or use a different column as needed
    data['High_Smoking_Prevalence'] = (data['scaler__Smoking_Prevalence'] > 0.5).astype(int)

    X = data.drop(columns=['scaler__Smoking_Prevalence', 'High_Smoking_Prevalence'])
    y = data['High_Smoking_Prevalence']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Logistic Regression Metrics")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='binary'))
    print("Recall:", recall_score(y_test, y_pred, average='binary'))

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    })
    feature_importance['Absolute_Coefficient'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values(by='Absolute_Coefficient', ascending=False)

    print("\nFeature Importance (Top 10):")
    print(feature_importance.head(10))


if __name__ == "__main__":
    run_logistic_regression()
