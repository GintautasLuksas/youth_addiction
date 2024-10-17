import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def evaluate_model(model, params, X_train, X_test, y_train, y_test):
    """
    Performs model evaluation with GridSearchCV, then displays Accuracy, Precision,
    and Recall metrics using a bar chart and a message box.
    """
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    }

    metrics = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, scores, color='skyblue')
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.show()

    result_text = "\n".join([f"{metric}: {score:.4f}" for metric, score in results.items()])
    messagebox.showinfo("Model Evaluation Results", result_text)


def choose_dataset(dataset_choice):
    """
    Loads the selected dataset based on the user's choice.
    """
    global X, y, data
    if dataset_choice == "Wine":
        data = load_wine()
    elif dataset_choice == "Diabetes":
        data = load_diabetes()
    elif dataset_choice == "Cancer":
        data = load_breast_cancer()

    X = data.data
    y = np.where(data.target > np.median(data.target), 1, 0) if data.target.ndim == 1 else data.target

    show_model_menu()


def configure_gridsearch(model_name):
    """
    Configures GridSearch parameters based on the selected model.
    """
    global model, params

    if model_name == "KNN":
        model = KNeighborsClassifier()
        params = {"n_neighbors": range(1, 21)}
        evaluate_gridsearch()

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        params = {"criterion": ["gini", "entropy"], "max_depth": range(1, 11)}
        evaluate_gridsearch()

    elif model_name == "Naive Bayes":
        model = GaussianNB()
        params = {}  # No GridSearch for Naive Bayes since it has no hyperparameters
        evaluate_model(model, params, *train_test_split(X, y, test_size=0.2, random_state=42))


def evaluate_gridsearch():
    """
    Splits the dataset, runs GridSearchCV on the selected model with specified parameters, and displays the results.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    evaluate_model(model, params, X_train, X_test, y_train, y_test)


def compare_all_methods(dataset_choice):
    """
    Automatically runs all three models (KNN, Decision Tree, Naive Bayes) on the selected dataset and compares Accuracy, Precision, and Recall.
    """
    global X, y
    if dataset_choice == "Wine":
        data = load_wine()
    elif dataset_choice == "Diabetes":
        data = load_diabetes()
    elif dataset_choice == "Cancer":
        data = load_breast_cancer()

    X = data.data
    y = np.where(data.target > np.median(data.target), 1, 0) if data.target.ndim == 1 else data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # KNN
    knn_params = {"n_neighbors": range(1, 21)}
    knn = KNeighborsClassifier()
    grid_knn = GridSearchCV(knn, knn_params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    knn_best = grid_knn.best_estimator_
    knn_accuracy = accuracy_score(y_test, knn_best.predict(X_test))
    knn_precision = precision_score(y_test, knn_best.predict(X_test), zero_division=0)
    knn_recall = recall_score(y_test, knn_best.predict(X_test), zero_division=0)
    results["KNN"] = (knn_accuracy, knn_precision, knn_recall)

    # Decision Tree
    tree_params = {"criterion": ["gini", "entropy"], "max_depth": range(1, 11)}
    decision_tree = DecisionTreeClassifier(random_state=42)
    grid_tree = GridSearchCV(decision_tree, tree_params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_tree.fit(X_train, y_train)
    tree_best = grid_tree.best_estimator_
    tree_accuracy = accuracy_score(y_test, tree_best.predict(X_test))
    tree_precision = precision_score(y_test, tree_best.predict(X_test), zero_division=0)
    tree_recall = recall_score(y_test, tree_best.predict(X_test), zero_division=0)
    results["Decision Tree"] = (tree_accuracy, tree_precision, tree_recall)

    # Naive Bayes
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    nb_accuracy = accuracy_score(y_test, naive_bayes.predict(X_test))
    nb_precision = precision_score(y_test, naive_bayes.predict(X_test), zero_division=0)
    nb_recall = recall_score(y_test, naive_bayes.predict(X_test), zero_division=0)
    results["Naive Bayes"] = (nb_accuracy, nb_precision, nb_recall)

    # Display comparison results
    comparison_text = "\n".join(
        [f"{model} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}"
         for model, (acc, prec, rec) in results.items()]
    )
    messagebox.showinfo("Comparison Results", comparison_text)

    # Plot comparison results
    methods = list(results.keys())
    accuracy_scores = [scores[0] for scores in results.values()]
    precision_scores = [scores[1] for scores in results.values()]
    recall_scores = [scores[2] for scores in results.values()]

    x = np.arange(len(methods))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, accuracy_scores, width, label="Accuracy")
    plt.bar(x, precision_scores, width, label="Precision")
    plt.bar(x + width, recall_scores, width, label="Recall")

    plt.xlabel("Models")
    plt.ylabel("Scores")
    plt.title(f"Model Comparison on {dataset_choice} Dataset")
    plt.xticks(x, methods)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def choose_dataset_for_comparison():
    """
    Displays the dataset selection menu for comparison purposes.
    """
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Select a Dataset for Comparison:", font=("Arial", 14)).pack(pady=10)

    tk.Button(root, text="Wine", command=lambda: compare_all_methods("Wine")).pack(pady=5)
    tk.Button(root, text="Diabetes", command=lambda: compare_all_methods("Diabetes")).pack(pady=5)
    tk.Button(root, text="Cancer", command=lambda: compare_all_methods("Cancer")).pack(pady=5)
    tk.Button(root, text="Back", command=show_main_menu).pack(pady=20)


def choose_dataset_for_separate():
    """
    Displays the dataset selection menu for separate dataset analysis (with GridSearch).
    """
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Select a Dataset:", font=("Arial", 14)).pack(pady=10)

    tk.Button(root, text="Wine", command=lambda: choose_dataset("Wine")).pack(pady=5)
    tk.Button(root, text="Diabetes", command=lambda: choose_dataset("Diabetes")).pack(pady=5)
    tk.Button(root, text="Cancer", command=lambda: choose_dataset("Cancer")).pack(pady=5)
    tk.Button(root, text="Back", command=show_main_menu).pack(pady=20)


def show_main_menu():
    """
    Displays the main menu with three options: Separate Datasets, Comparison, Exit.
    """
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Main Menu", font=("Arial", 16)).pack(pady=10)

    tk.Button(root, text="Separate Datasets", command=choose_dataset_for_separate).pack(pady=10)
    tk.Button(root, text="Comparison", command=choose_dataset_for_comparison).pack(pady=10)
    tk.Button(root, text="Exit", command=root.quit).pack(pady=10)


def show_model_menu():
    """
    Displays the model selection menu within the same Tkinter window.
    """
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Select a Model:", font=("Arial", 14)).pack(pady=10)

    tk.Button(root, text="K-Nearest Neighbors", command=lambda: configure_gridsearch("KNN")).pack(pady=5)
    tk.Button(root, text="Decision Tree", command=lambda: configure_gridsearch("Decision Tree")).pack(pady=5)
    tk.Button(root, text="Naive Bayes", command=lambda: configure_gridsearch("Naive Bayes")).pack(pady=5)

    tk.Button(root, text="Back", command=show_main_menu).pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Machine Learning Model Selection")
    show_main_menu()
    root.mainloop()
