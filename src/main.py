# src/main.py
import logistic_regression
import decision_tree
import random_forest
import knn
import naive_bayes


def main():
    print("Choose a model to run:")
    print("1. Logistic Regression")
    print("2. Decision Tree")
    print("3. Random Forest")
    print("4. K-Nearest Neighbors (KNN)")
    print("5. Naive Bayes")
    print("6. Exit")

    choice = input("Enter the number of your choice: ")

    if choice == '1':
        logistic_regression.run_logistic_regression()
    elif choice == '2':
        decision_tree.run_decision_tree()
    elif choice == '3':
        random_forest.run_random_forest()
    elif choice == '4':
        knn.run_knn()
    elif choice == '5':
        naive_bayes.run_naive_bayes()
    elif choice == '6':
        print("Exiting the program.")
    else:
        print("Invalid choice. Please enter a number from 1 to 6.")


if __name__ == "__main__":
    main()
